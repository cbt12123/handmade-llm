"""Hand-authored numerical vectors, not learned semantic representations."""
import math
from common import save


def cosine(a,b):
    if len(a) != len(b) or not a:
        raise ValueError('维度必须相等且非空')
    denominator = math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    return sum(x*y for x,y in zip(a,b))/denominator if denominator else 0.0


if __name__ == '__main__':
    query = [1,2]
    documents = {'A':[2,4], 'B':[2,0], 'C':[-1,-2]}
    scores = {key:cosine(query,value) for key,value in documents.items()}
    assert math.isclose(scores['A'],1) and math.isclose(scores['C'],-1)
    save('24_vectors.json',{'query':query,'documents':documents,'cosine':scores,
                           'scope':'Manually specified vectors illustrate cosine; not semantic retrieval.'})
    print(scores)
