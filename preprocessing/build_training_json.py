"""
Build contrastive training JSON from annotated XML-to-JSON output.

For each case JSON in demosthenes_dataset_json/, extract chains of related spans,
assemble suffix strings with TYPE_<TAG> markup, and create training examples
where one span is masked (<mask>) and the positive is the masked span's original markup.
Also collect all sentences as the target set.
"""
import os
import json
import glob
import re
try:
    from spacy.lang.en import English
    _USE_SPACY = True
    _nlp = English()
    _nlp.add_pipe('sentencizer')
except ImportError:
    _USE_SPACY = False
    _nlp = None


def load_annotations(json_path):
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    plain = data['document'].get('plainText', '')
    anns = data.get('annotations', [])
    # Get the doc name from name or fall back to filename
    doc_name = data['document'].get('name') or os.path.splitext(os.path.basename(json_path))[0]
    return plain, anns, doc_name


def build_chain_map(annotations):
    # map id -> annotation dict
    id2ann = {ann['_id']: ann for ann in annotations if ann.get('_id')}
    # build child links using SUP attribute as relation
    children = {}
    for ann in annotations:
        sup = ann.get('attributes', {}).get('SUP')
        if not sup:
            continue
        sups = sup if isinstance(sup, list) else [sup]
        for p in sups:
            children.setdefault(p, []).append(ann['_id'])
    # root annotations: those without SUP
    roots = [i for i, ann in id2ann.items() if not ann['attributes'].get('SUP')]
    chains = []
    for root in roots:
        chain = [root]
        while True:
            nxts = children.get(chain[-1], [])
            if not nxts:
                break
            nxt = nxts[0]
            if nxt in chain:
                break
            chain.append(nxt)
        if len(chain) > 1:
            chains.append(chain)
    return chains, id2ann


def markup_span(ann, text):
    # wrap span text in TYPE_<S> tag with id, SUP->rel, and subtype if present
    span = text[ann['start']:ann['end']]
    attrs = ann.get('attributes', {})
    # Determine initial type (lowercase): conclusion vs. legal/factual
    if ann.get('name') == 'conc':
        type_key = 'conclusion'
    else:
        tval = attrs.get('T')
        if tval == 'L':
            type_key = 'legal'
        elif tval == 'F':
            type_key = 'factual'
        elif tval in ('L|F', 'F|L'):
            type_key = 'legal_and_factual'
        else:
            type_key = ann.get('name', '')

    parts = [f"TYPE_{type_key}"]
    if ann.get('_id'):
        parts.append(f"id={ann['_id']}")

    # S attribute gives semantic tag(s)
    # Map S attribute to Argument_scheme values
    sval = attrs.get('S')
    if sval:
        s_list = sval if isinstance(sval, list) else [sval]
        schemes = []
        for s in s_list:
            if s == 'Prec': schemes.append('Argument from Precedent')
            elif s == 'Aut': schemes.append('Authoritative Argument')
            elif s == 'Class': schemes.append('Argument from Verbal Classification')
            elif s == 'Itpr': schemes.append('Argument from Interpretation')
            elif s == 'Princ': schemes.append('Argument from Principle')
        if schemes:
            parts.append(f"Argument_scheme={'|'.join(schemes)}")

    # Include SUP links as antecedents
    sup = attrs.get('SUP')
    if sup:
        sup_list = sup if isinstance(sup, list) else [sup]
        parts.append(f"antecedents={'|'.join(sup_list)}")

    open_tag = '<' + ' '.join(parts) + '>'
    close_tag = f'</TYPE_{type_key}>'
    return open_tag + span + close_tag


def split_sentences(text):
    """
    Segment text into coherent chunks:
      - Headings (line without ending punctuation, in Title/ALL CAPS) become standalone.
      - Lists: intro line ending ':' followed by bullets ('a. ...'), each bullet prefixed with intro.
      - Remaining paragraphs are sentence-segmented via spaCy's sentencizer.
    """
    # use spaCy sentencizer if available, else fallback to regex on punctuation+uppercase
    nlp = _nlp if _USE_SPACY else None

    lines = text.splitlines()
    chunks = []
    i = 0
    heading_re = re.compile(r'^[A-Z][A-Za-z0-9 ,&\-]+$')
    bullet_re = re.compile(r'^[a-z]\.')
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Heading without trailing punctuation
        if heading_re.match(line) and line[-1] not in '.?!:':
            chunks.append(line)
            i += 1
            continue

        # List intro + bullets
        if line.endswith(':') and i + 1 < len(lines) and bullet_re.match(lines[i+1].strip()):
            intro = line.rstrip(':').strip()
            i += 1
            while i < len(lines) and bullet_re.match(lines[i].strip()):
                item = lines[i].strip()
                chunks.append(f"{intro} {item}")
                i += 1
            continue

        # Paragraph -> spaCy sentences
        para = [line]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt or heading_re.match(nxt) or nxt.endswith(':'):
                break
            para.append(nxt)
            j += 1
        para_text = ' '.join(para)
        if nlp:
            for sent in nlp(para_text).sents:
                s = sent.text.strip()
                if s:
                    chunks.append(s)
        else:
            # fallback: split on punctuation followed by uppercase
            pattern = r'(?<=[\.\?\!])\s+(?=[A-Z])'
            for s in re.split(pattern, para_text):
                s2 = s.strip()
                if s2:
                    chunks.append(s2)
        i = j

    return chunks


def build_training_examples(plain, chains, id2ann, doc_id):
    examples = []
    for chain in chains:
        # markup full chain
        marked = [markup_span(id2ann[i], plain) for i in chain]
        # for each position, mask it
        for idx, span_markup in enumerate(marked):
            prefix = marked.copy()
            prefix[idx] = '[MASK]'

            ann = id2ann[chain[idx]]
            pos_text = plain[ann['start']:ann['end']]

            examples.append({
                'prefix': ' '.join(prefix),
                'positive': pos_text,
                'doc_id': doc_id,
            })
    return examples


def main():
    base = os.path.dirname(__file__)
    src_dir = os.path.join(base, 'demosthenes_dataset_json')
    out_dir = os.path.join(base, 'demosthenes_training_json')
    os.makedirs(out_dir, exist_ok=True)

    for path in glob.glob(os.path.join(src_dir, '*.json')):
        plain, anns, doc_id = load_annotations(path)
        chains, id2ann = build_chain_map(anns)
        examples = build_training_examples(plain, chains, id2ann, doc_id)
        targets = split_sentences(plain)

        fname = os.path.splitext(os.path.basename(path))[0] + '_train.json'
        out_path = os.path.join(out_dir, fname)
        with open(out_path, 'w', encoding='utf8') as f:
            json.dump({'examples': examples, 'target_set': targets}, f,
                      ensure_ascii=False, indent=2)

if __name__ == '__main__':
    import re
    main()
