import fasttext
import fasttext.util
from fasttext import load_model

classifier = load_model(r"model1.bin")

def print_results(N, p, r):
    print("N\t" + str(N))
    print("precision", "P@{}\t{:.3f}".format(1, p))
    print("recall", "R@{}\t{:.3f}".format(1, r))


# To evaluate our model by computing the precision at 1 (P@1) and the recall on a test set, we use the test function:
print_results(*classifier.test(r'D:\WFH\untitled\venv\march2022 data\test.txt'))


