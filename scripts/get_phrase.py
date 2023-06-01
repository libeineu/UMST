import sys
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def get_dp(s):
    s = s.strip()
    s = s.replace(' ', '')
    string = s.split(",")
    result = []
    for i in range(len(string)):
        if is_number(string[i]):
            result.append(string[i])
        elif len(string[i]) == 1 and is_number(string[i]):
            result.append(string[i])
        elif len(string[i]) != 1 and is_number(string[i][:-1]):
            result.append(string[i][:-1])
        elif len(string[i]) != 1 and is_number(string[i][:-2]):
            result.append(string[i][:-2])
    return " ".join(result)

input_file = open(sys.argv[1],"r")
output_file = open(sys.argv[2],"w")

for line in input_file:
    result = get_dp(line)
    output_file.write(result+"\n")