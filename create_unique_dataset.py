import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FILENAME_LEGAL_EN = 'data/legal.en'
FILENAME_LEGAL_SPARQL = 'data/legal.sparql'
FILENAME_UNIQUE_EN = 'data/unique.legal.en'
FILENAME_UNIQUE_SPARQL = 'data/unique.legal.sparql'

def read_file(filename):
    if os.path.exists(filename):
        with open(filename) as f:
            lines = f.readlines()
    return lines

# function to get unique values
def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

def save(list1, filename):
    with open(filename, 'w') as f:
        for item in list1:
            f.write("%s" % item)

save(unique(read_file(FILENAME_LEGAL_EN)),FILENAME_UNIQUE_EN)
save(unique(read_file(FILENAME_LEGAL_SPARQL)),FILENAME_UNIQUE_SPARQL)
