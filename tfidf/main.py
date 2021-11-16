
def clean(data, col):

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)' ,r'\1 \2 \3')
    # Replace repeating characters
    data[col] = data[col].str.replace(r'(")\1+' ,r'\1')
    data[col] = data[col].str.replace(r'([*!?\'])\1\1+\B' ,r'\1\1')
    data[col] = data[col].str.replace(r'(\w)\1\1+\B' ,r'\1\1')
    data[col] = data[col].str.replace(r'(\w)\1+\b' ,r'\1').str.strip()

    return data