from stanfordcorenlp import StanfordCoreNLP
langs = ['en']
lang='en'
nlp_lang0 = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', lang=langs[0])

data_modes = ['train','test','valid']

basic_input_path = 'path/to/raw-data(before bpe)'
basic_output_path = 'path/to/output file'

for mode in data_modes:
    input_path = basic_input_path + '/' + mode + '.' + lang
    output_path = basic_output_path + '/' + mode + '.' + lang
    input_file = open(input_path, 'r', encoding='utf-8')
    output_file = open(output_path, 'w', encoding='utf-8')

    for index, line in enumerate(input_file.readlines()):
        output_file.write(str(nlp_lang0.dependency_parse(line.strip()))+'\n')
        if index % 2000 == 0:
            print(output_path + str(index))
    input_file.close()
    output_file.close()


nlp_lang0.close()