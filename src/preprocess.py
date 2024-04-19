def preprocess(text, lower=True, remove_punc=True, remove_digit=True, stem='english'):
    import string
    import re
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    res = text
    
    if lower:
        res = res.lower()

    if remove_punc:
        punc = string.punctuation
        res = res.translate(str.maketrans(punc, ' ' * len(punc)))

    if remove_digit:
        res = re.sub('[0-9]+', '', res)

    if stem:
        after_stem = ''
        stemmer = SnowballStemmer(stem, ignore_stopwords=True)
        for token in res.split(' '):
                after_stem += stemmer.stem(token) + ' '

        res = after_stem

    return ' '.join(word_tokenize(res))

