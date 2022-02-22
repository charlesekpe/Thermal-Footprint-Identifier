import re

def extract_first_number(string):
    '''
    Returns the first number in a string
    @Parameters
        string: string
            string to search numbers in
    @Returns:
        int
            the first number detected        
    '''
    nums = [int(s) for s in re.findall(r'\d+', string)]
    if nums != []:
        return nums[0]
    else:
        print('No number detected.')
        return nums