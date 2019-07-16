# -*-coding:utf-8-*-
def iob_ranges(words, tags):
    """
    IOB -> Ranges
    """
    begin = None
    assert len(words) == len(tags)
    ranges = []

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split('-')[0] == 'O':
            ranges.append({
                'entity': ''.join(words[begin: i + 1]),
                'type': type2chinese(temp_type),
                # 'start': begin,
                # 'end': i
            })

    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'O':
            pass
        elif tag.split('-')[0] == 'B':
            begin = i
            temp_type = tag.split('-')[1]
            check_if_closing_range()
        elif tag.split('-')[0] == 'I':
            if begin is None:
                # print(tags)
                begin = i
                temp_type = tag.split('-')[1]
            check_if_closing_range()
    return ranges


def type2chinese(type):
    if type == "E":
        return "举证方"
    if type == "O":
        return "质证意见"
    if type == "A":
        return "质证方"
    if type == "T":
        return "证据名称"
    if type == "C":
        return "证明内容"
