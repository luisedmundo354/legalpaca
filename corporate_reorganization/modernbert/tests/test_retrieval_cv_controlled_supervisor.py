from __future__ import annotations

import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    aws,
    controlled_supervisor,
    determinism_gate,
)
from corporate_reorganization.modernbert.tests.test_retrieval_cv_training_aws import (
    _staging_receipt,
    _training_plan,
)


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


def _seal(payload: dict[str, object]) -> dict[str, object]:
    receipt = copy.deepcopy(payload)
    receipt["receipt_sha256"] = _document_sha256(payload)
    return receipt


def _determinism_gate_receipt(
    plan: dict[str, object], staged: dict[str, object]
) -> dict[str, object]:
    return _seal(
        {
            "schema_version": 3,
            "protocol": determinism_gate.DETERMINISM_GATE_PROTOCOL,
            "plan_sha256": _document_sha256(plan),
            "staging_receipt_sha256": _document_sha256(staged),
            "exact_match": True,
        }
    )


def _identity_validator(value: object, **_: object) -> object:
    return copy.deepcopy(value)


class _RecordedSageMaker:
    def __init__(self, harness: "_LaunchHarness") -> None:
        self._harness = harness

    def create_training_job(self, **request: object) -> dict[str, str]:
        run_id = request.get("SupervisorRunId")
        if type(run_id) is not str:
            raise AssertionError("Synthetic request lost its supervisor run ID")
        if run_id in self._harness.created_run_ids:
            raise AssertionError(f"Duplicate CreateTrainingJob for {run_id}")
        self._harness.created_run_ids.append(run_id)
        self._harness.active_run_ids.add(run_id)
        self._harness.maximum_active = max(
            self._harness.maximum_active,
            len(self._harness.active_run_ids),
        )
        self._harness.events.append(("create", run_id))
        return {"TrainingJobArn": f"arn:aws:sagemaker:test:job/{run_id}"}


class _LaunchHarness:
    def __init__(self, plan: dict[str, object]) -> None:
        self.runs = {
            run["run_id"]: run
            for run in (*plan["controlled_runs"], *plan["auxiliary_runs"])
        }
        self.created_run_ids: list[str] = []
        self.active_run_ids: set[str] = set()
        self.maximum_active = 0
        self.events: list[tuple[str, str]] = []
        self.status_by_run_id: dict[str, str] = {}
        self.default_status = "InProgress"
        self.raise_after_create: BaseException | None = None
        self.raise_before_create: BaseException | None = None
        self.rerun_preflight_in_submit = False
        self.preflight_active_counts: list[int] = []
        self.sagemaker = _RecordedSageMaker(self)
        self.clients = aws.AwsClients(
            sts=Mock(),
            iam=Mock(),
            ecr=Mock(),
            s3=Mock(),
            service_quotas=Mock(),
            ec2=Mock(),
            sagemaker=self.sagemaker,
            logs=Mock(),
        )

    def preflight_training_job(
        self,
        _clients: aws.AwsClients,
        *,
        run_id: str,
        **_: object,
    ) -> dict[str, object]:
        run = self.runs[run_id]
        request = {
            "TrainingJobName": run["job_name"],
            "SupervisorRunId": run_id,
        }
        request_receipt = {
            "request": request,
            "request_sha256": _document_sha256(request),
        }
        active_count = (
            self.preflight_active_counts.pop(0)
            if self.preflight_active_counts
            else 0
        )
        self.events.append(("preflight", run_id))
        return _seal(
            {
                "run_id": run_id,
                "job_name": run["job_name"],
                "request_receipt": request_receipt,
                "active_planned_jobs": {
                    "count": active_count,
                    "job_names": [f"synthetic-active-{index}" for index in range(active_count)],
                },
            }
        )

    def submit_training_job_once(
        self,
        clients: aws.AwsClients,
        *,
        preflight_receipt: dict[str, object],
        training_plan: dict[str, object],
        staging_receipt: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        run_id = preflight_receipt["run_id"]
        if self.rerun_preflight_in_submit:
            fresh = self.preflight_training_job(
                clients,
                training_plan=training_plan,
                staging_receipt=staging_receipt,
                run_id=run_id,
            )
            if aws.canonical_json_bytes(fresh) != aws.canonical_json_bytes(
                preflight_receipt
            ):
                raise RuntimeError(
                    "Fresh preflight differs from the approved saved receipt"
                )
        if self.raise_before_create is not None:
            raise self.raise_before_create
        request = preflight_receipt["request_receipt"]["request"]
        response = clients.sagemaker.create_training_job(**copy.deepcopy(request))
        if self.raise_after_create is not None:
            raise self.raise_after_create
        self.events.append(("submission", run_id))
        return _seal(
            {
                "run_id": run_id,
                "job_name": preflight_receipt["job_name"],
                "job_arn": response["TrainingJobArn"],
            }
        )

    def describe_training_job_status(
        self,
        _clients: aws.AwsClients,
        *,
        submission_receipt: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        run_id = submission_receipt["run_id"]
        remote_status = self.status_by_run_id.get(run_id, self.default_status)
        self.events.append(("status", run_id))
        return _seal(
            {
                "run_id": run_id,
                "snapshot": {"training_job_status": remote_status},
            }
        )

    def verify_terminal_training_job(
        self,
        _clients: aws.AwsClients,
        *,
        submission_receipt: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        run_id = submission_receipt["run_id"]
        terminal_status = self.status_by_run_id.get(run_id, self.default_status)
        self.active_run_ids.remove(run_id)
        self.events.append(("terminal", run_id))
        return _seal(
            {
                "run_id": run_id,
                "terminal_status": terminal_status,
                "succeeded": terminal_status == "Completed",
            }
        )


class ControlledTrainingSupervisorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.plan, _ = _training_plan(self.root / "plan")
        self.staging = _staging_receipt(self.plan)
        self.gate = _determinism_gate_receipt(self.plan, self.staging)
        self.state_dir = self.root / "controlled-supervisor"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @contextmanager
    def _runtime(self, harness: _LaunchHarness):
        with (
            patch.object(
                controlled_supervisor.determinism_gate,
                "validate_determinism_gate_receipt",
                side_effect=_identity_validator,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "preflight_training_job",
                side_effect=harness.preflight_training_job,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "submit_training_job_once",
                side_effect=harness.submit_training_job_once,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "describe_training_job_status",
                side_effect=harness.describe_training_job_status,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "verify_terminal_training_job",
                side_effect=harness.verify_terminal_training_job,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "validate_training_preflight_receipt",
                side_effect=_identity_validator,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "validate_training_submission_receipt",
                side_effect=_identity_validator,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "validate_training_status_receipt",
                side_effect=_identity_validator,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "validate_training_terminal_receipt",
                side_effect=_identity_validator,
            ),
        ):
            yield

    def _initialize(self) -> dict[str, object]:
        return controlled_supervisor.initialize_controlled_supervisor_state(
            state_dir=self.state_dir,
            training_plan=self.plan,
            staging_receipt=self.staging,
            determinism_gate_receipt=self.gate,
        )

    def test_gate_is_required_before_state_publication_and_schedule_is_exact(self) -> None:
        with patch.object(
            controlled_supervisor.determinism_gate,
            "validate_determinism_gate_receipt",
            side_effect=ValueError("synthetic determinism mismatch"),
        ):
            with self.assertRaisesRegex(ValueError, "determinism mismatch"):
                self._initialize()
        self.assertFalse(self.state_dir.exists())

        old_gate = copy.deepcopy(self.gate)
        old_gate["schema_version"] = 2
        old_gate = _seal(
            {
                key: value
                for key, value in old_gate.items()
                if key != "receipt_sha256"
            }
        )
        with patch.object(
            controlled_supervisor.determinism_gate,
            "validate_determinism_gate_receipt",
            side_effect=_identity_validator,
        ):
            with self.assertRaisesRegex(ValueError, "matching sealed v3 gate"):
                controlled_supervisor.initialize_controlled_supervisor_state(
                    state_dir=self.state_dir,
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    determinism_gate_receipt=old_gate,
                )
        self.assertFalse(self.state_dir.exists())

        with patch.object(
            controlled_supervisor.determinism_gate,
            "validate_determinism_gate_receipt",
            side_effect=_identity_validator,
        ) as validate_gate:
            supervisor = self._initialize()
        validate_gate.assert_called_once()

        schedule = supervisor["schedule"]
        self.assertEqual(len(schedule), 60)
        self.assertEqual(
            [entry["run_id"] for entry in schedule[:8]],
            [
                "controlled-f0-flat-local-s17",
                "controlled-f0-flat-global-s17",
                "controlled-f0-struct-local-s17",
                "controlled-f0-struct-global-s17",
                "controlled-f0-flat-local-s29",
                "controlled-f0-flat-global-s29",
                "controlled-f0-struct-local-s29",
                "controlled-f0-struct-global-s29",
            ],
        )
        for quartet_index in range(15):
            quartet = schedule[quartet_index * 4 : quartet_index * 4 + 4]
            self.assertEqual(
                {(entry["outer_fold"], entry["experiment_seed"]) for entry in quartet},
                {(quartet[0]["outer_fold"], quartet[0]["experiment_seed"])},
            )
            self.assertEqual(
                [(entry["query_view"], entry["sampler"]) for entry in quartet],
                [
                    ("flat_masked", "local_unique"),
                    ("flat_masked", "global_uniform"),
                    ("structured", "local_unique"),
                    ("structured", "global_uniform"),
                ],
            )
        auxiliary_ids = {run["run_id"] for run in self.plan["auxiliary_runs"]}
        self.assertTrue(
            auxiliary_ids.isdisjoint(entry["run_id"] for entry in schedule)
        )

    def test_first_quartet_and_restart_do_not_duplicate_creates(self) -> None:
        harness = _LaunchHarness(self.plan)
        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            snapshot = supervisor.advance_once()
            self.assertEqual(snapshot["counts"]["active"], 4)
            self.assertEqual(
                harness.created_run_ids,
                [
                    "controlled-f0-flat-local-s17",
                    "controlled-f0-flat-global-s17",
                    "controlled-f0-struct-local-s17",
                    "controlled-f0-struct-global-s17",
                ],
            )
            for entry in supervisor._supervisor["schedule"][:4]:
                run_dir = self.state_dir / "runs" / (
                    f"{entry['queue_index']:02d}-{entry['run_id']}"
                )
                self.assertTrue((run_dir / "create-intent.json").is_file())
                self.assertTrue((run_dir / "submission.json").is_file())
                intent, _ = (
                    controlled_supervisor.strict_config.load_canonical_json_object(
                        run_dir / "create-intent.json"
                    )
                )
                self.assertEqual(
                    intent["preflight_receipt"]["run_id"],
                    entry["run_id"],
                )

            restarted = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            restarted_snapshot = restarted.advance_once()
            self.assertEqual(restarted_snapshot["counts"]["active"], 4)
            self.assertEqual(len(harness.created_run_ids), 4)
            for entry in restarted._supervisor["schedule"][:4]:
                run_dir = self.state_dir / "runs" / (
                    f"{entry['queue_index']:02d}-{entry['run_id']}"
                )
                self.assertTrue(
                    (run_dir / "observations" / "status-000001.json").is_file()
                )

    def test_one_completion_backfills_the_next_scheduled_run_immediately(self) -> None:
        harness = _LaunchHarness(self.plan)
        completed = "controlled-f0-flat-local-s17"
        fifth = "controlled-f0-flat-local-s29"
        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            supervisor.advance_once()
            harness.status_by_run_id[completed] = "Completed"
            snapshot = supervisor.advance_once()

        self.assertEqual(snapshot["counts"]["completed"], 1)
        self.assertEqual(snapshot["counts"]["active"], 4)
        self.assertEqual(harness.created_run_ids[4], fifth)
        self.assertEqual(len(harness.created_run_ids), 5)
        self.assertLess(
            harness.events.index(("terminal", completed)),
            harness.events.index(("create", fifth)),
        )
        first_run = self.state_dir / "runs" / f"00-{completed}"
        self.assertTrue((first_run / "terminal.json").is_file())

    def test_create_without_submission_is_permanently_ambiguous(self) -> None:
        harness = _LaunchHarness(self.plan)
        harness.raise_after_create = TimeoutError("synthetic lost create response")
        first = "controlled-f0-flat-local-s17"
        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            with self.assertRaisesRegex(TimeoutError, "lost create response"):
                supervisor.advance_once()
            first_run = self.state_dir / "runs" / f"00-{first}"
            self.assertTrue((first_run / "create-intent.json").is_file())
            self.assertFalse((first_run / "submission.json").exists())
            with self.assertRaisesRegex(
                RuntimeError, "refusing retry or reconciliation"
            ):
                controlled_supervisor.ControlledTrainingSupervisor(
                    harness.clients, state_dir=self.state_dir
                )
        self.assertEqual(harness.created_run_ids, [first])

    def test_dynamic_preflight_mismatch_is_refreshable_on_restart(self) -> None:
        harness = _LaunchHarness(self.plan)
        harness.rerun_preflight_in_submit = True
        harness.preflight_active_counts = [3, 2]
        first = "controlled-f0-flat-local-s17"
        first_run = self.state_dir / "runs" / f"00-{first}"
        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            with self.assertRaisesRegex(RuntimeError, "Fresh preflight differs"):
                supervisor.advance_once()
            self.assertFalse((first_run / "preflight.json").exists())
            self.assertFalse((first_run / "create-intent.json").exists())
            self.assertFalse((first_run / "submission.json").exists())
            self.assertFalse(harness.created_run_ids)

            harness.preflight_active_counts = [0, 0]
            restarted = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            snapshot = restarted.advance_once()
        self.assertEqual(snapshot["counts"]["active"], 4)
        self.assertEqual(harness.created_run_ids[0], first)

    def test_exception_before_intent_can_restart_without_stale_preflight(self) -> None:
        harness = _LaunchHarness(self.plan)
        harness.raise_before_create = RuntimeError("synthetic pre-intent crash")
        first = "controlled-f0-flat-local-s17"
        first_run = self.state_dir / "runs" / f"00-{first}"
        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            with self.assertRaisesRegex(RuntimeError, "pre-intent crash"):
                supervisor.advance_once()
            self.assertEqual(list(first_run.iterdir()), [first_run / "observations"])
            self.assertFalse(harness.created_run_ids)

            harness.raise_before_create = None
            restarted = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            restarted.advance_once()
        self.assertEqual(harness.created_run_ids[0], first)

    def test_run_directory_swap_before_publication_prevents_create(self) -> None:
        harness = _LaunchHarness(self.plan)
        first = "controlled-f0-flat-local-s17"
        first_run = self.state_dir / "runs" / f"00-{first}"
        displaced = self.root / f"{first_run.name}.displaced"
        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            first_run.rename(displaced)
            first_run.mkdir()
            (first_run / "observations").mkdir()
            with self.assertRaisesRegex(RuntimeError, "path identity changed"):
                supervisor.advance_once()
        self.assertFalse(harness.created_run_ids)

    def test_run_directory_swap_after_link_prevents_create(self) -> None:
        harness = _LaunchHarness(self.plan)
        first = "controlled-f0-flat-local-s17"
        first_run = self.state_dir / "runs" / f"00-{first}"
        displaced = self.root / f"{first_run.name}.displaced"
        real_link = controlled_supervisor.os.link
        swapped = False

        def link_then_swap(
            source: str,
            target: str,
            **kwargs: object,
        ) -> None:
            nonlocal swapped
            real_link(source, target, **kwargs)
            if target == "create-intent.json" and not swapped:
                swapped = True
                first_run.rename(displaced)
                first_run.mkdir()
                (first_run / "observations").mkdir()

        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            with patch.object(
                controlled_supervisor.os,
                "link",
                side_effect=link_then_swap,
            ):
                with self.assertRaisesRegex(RuntimeError, "path identity changed"):
                    supervisor.advance_once()
        self.assertTrue(swapped)
        self.assertFalse(harness.created_run_ids)
        self.assertFalse((displaced / "create-intent.json").exists())
        self.assertFalse((first_run / "create-intent.json").exists())

    def test_terminal_failure_is_persisted_and_stops_before_backfill(self) -> None:
        harness = _LaunchHarness(self.plan)
        failed = "controlled-f0-flat-local-s17"
        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            supervisor.advance_once()
            harness.status_by_run_id[failed] = "Failed"
            with self.assertRaisesRegex(RuntimeError, "ended unsuccessfully"):
                supervisor.advance_once()
            failed_run = self.state_dir / "runs" / f"00-{failed}"
            terminal, _ = controlled_supervisor.strict_config.load_canonical_json_object(
                failed_run / "terminal.json"
            )
            self.assertEqual(terminal["terminal_status"], "Failed")
            with self.assertRaisesRegex(RuntimeError, "terminal failure evidence"):
                controlled_supervisor.ControlledTrainingSupervisor(
                    harness.clients, state_dir=self.state_dir
                )
        self.assertEqual(len(harness.created_run_ids), 4)

    def test_complete_pipeline_launches_exactly_60_with_four_active(self) -> None:
        harness = _LaunchHarness(self.plan)
        harness.default_status = "Completed"
        with self._runtime(harness), patch.object(
            controlled_supervisor.time, "sleep"
        ) as sleep:
            initialized = self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            snapshot = supervisor.run_until_complete(poll_interval_seconds=1)
            created_before_restart = list(harness.created_run_ids)
            restarted = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            restarted_snapshot = restarted.advance_once()

        self.assertTrue(snapshot["complete"])
        self.assertEqual(restarted_snapshot, snapshot)
        self.assertEqual(len(harness.created_run_ids), 60)
        self.assertEqual(
            created_before_restart,
            [entry["run_id"] for entry in initialized["schedule"]],
        )
        self.assertEqual(harness.maximum_active, 4)
        self.assertFalse(harness.active_run_ids)
        self.assertGreater(sleep.call_count, 0)
        completion, _ = controlled_supervisor.strict_config.load_canonical_json_object(
            self.state_dir / "completion.json"
        )
        self.assertEqual(
            completion["protocol"],
            controlled_supervisor.CONTROLLED_SUPERVISOR_COMPLETION_PROTOCOL,
        )
        self.assertEqual(completion["completed_runs"], 60)
        auxiliary_ids = {run["run_id"] for run in self.plan["auxiliary_runs"]}
        self.assertTrue(auxiliary_ids.isdisjoint(harness.created_run_ids))

    def test_poll_interval_is_explicit_and_strict(self) -> None:
        harness = _LaunchHarness(self.plan)
        with self._runtime(harness):
            self._initialize()
            supervisor = controlled_supervisor.ControlledTrainingSupervisor(
                harness.clients, state_dir=self.state_dir
            )
            for invalid in (0, -1, True, 1.0):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "positive exact integer"):
                        supervisor.run_until_complete(
                            poll_interval_seconds=invalid
                        )
        self.assertFalse(harness.created_run_ids)


if __name__ == "__main__":
    unittest.main()
