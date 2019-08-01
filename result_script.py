# -*-coding:utf-8-*-
import codecs


def analyze_result(type):
    with codecs.open("./data/test.tsv", 'r', 'utf-8') as r:
        text = r.readlines()
    import re
    result = []
    with codecs.open("cl_result.txt", 'r', 'utf-8') as r:
        for index, line in enumerate(r.readlines()):
            if re.search(r"^" + type + "\t[^" + type + "]", line):
                context = line.split("\t")
                temp = [context[0], context[1], text[index].split("\t")[1]]
                result.append(temp)
    with codecs.open("%s_result.txt" % type, "w", "utf-8") as w:
        [w.write("\t".join(r) + "\n") for r in result]


if __name__ == '__main__':
    analyze_result('V')
    analyze_result("E")
