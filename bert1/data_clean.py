import re


def remove_space_links(string):
    #     string = BeautifulSoup(string).text.strip().lower()
    string = re.sub(r'((http)\S+)', 'http', string)
    string = re.sub(r'\s+', ' ', string)
    return string


def remove_numbers(x):
    x = re.sub('\d+', ' ', x)
    return x


def preprocessText(input_str):
    # convert text to lowercase
    input_str = input_str.lower()
    # # Remove punctuation
    # input_str = input_str.translate(str.maketrans('', '', string.punctuation))

    # repalce change lines
    input_str = input_str.replace("\r", " ")
    input_str = input_str.replace("\n", " ")
    input_str = input_str.replace("�", " ")

    #
    input_str = re.sub('[\W_]+', ' ', input_str)

    # remove white space
    output_str = re.sub(' +', ' ', input_str)
    output_str = output_str.strip()
    return output_str


def convert_str(x):
    return str(x)


def clean_data(df, cols: list):
    for col in cols:
        df[col] = df[col].apply(lambda x: convert_str(x))
        df[col] = df[col].apply(lambda x: preprocessText(x))
        df[col] = df[col].apply(lambda x: remove_space_links(x))
        df[col] = df[col].apply(lambda x: remove_numbers(x))

    return df