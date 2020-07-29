import numpy as np
from esig import tosig as ts
import math
import random
import itertools

def prune (path):
    """
    Function prunes tree-like components of a path, passing a single time over the list of increments. The input needs to be very specific. In particular, linearly dependant increments need to be either equal or inverse. For exhaustive pruning, use 'fullprune'. In its use in 'fullprune' the increments have supremum norm 1.

    Args:
        path (list): path given as a list of increments.
       

    Returns:
        path (list): pruned path as a list of increment vectors. Numerical values will be of type np.float64 or np.int32

    """
    path=np.array(path)
   
    l=len(path)
    copy = path-path
    it=iter(range (l))
    array=path

    for i in it:
        if i<l-1:    
            if any(array[i] + array[i+1]): 
                copy[i] = array[i]
            else: i=next(it)
        else: copy[i] = array[i]  
    copy=np.array(copy)  
    copy=list(copy)
    copy=[list(x) for x in copy]
    copy=[x for x in copy if any(x)]
    #copy=np.array(copy)
      
    return copy


def fullprune (path, o=None, x_0=None):
    """

    Function loops 'prune' to exhaustively prune the input path. It refines the path to ensure that the use of 'prune' is made with the correct type of input. 

    Args:
        path (list): path given either as a list of position vectors or list of increments.
        o (bool): optional parameter, if 'True', x_0 is required and the input is treated as increments
        x_0 (list): position vector, specifying initial point of the path if 'o==True'

    Returns:
        path (list): pruned path as a list of position vectors. Numerical values will be of type np.float64 or np.int32

    """
    

    if x_0 and o is None: 
        raise ValueError("Set o=1!")


    if o==None:
        first=path[0]
    else:
        first=x_0


    if o is None:
        # turn path into increments
        
        l=len(path)
       
        array=np.array(path)
        incr=array-array
        for i in range(l-1):
             incr[i]=array[i+1]-array[i]

    # refine the path

    dim=len(incr[0])
    length=len(incr)
    new_length=0
    list_lengths=[]
    for i in range(length):
        new_length += np.int32(max(np.abs(incr[i])))
        list_lengths.append(np.int32(max(np.abs(incr[i]))))
    refined_path=np.zeros((new_length,dim))
    counter=0
    for i in range(length):
        for j in range(list_lengths[i]):

            refined_path[counter]=(1/ list_lengths[i])*incr[i]

            counter=counter+1

    #print(refined_path)
    incr=refined_path
    l_in=len(incr)
    #print(l_in)
    l_out=0
    while l_in > l_out:
        l_in=len(incr)
        incr=prune(incr)
        l_out=len(incr)
    #print(l_out)
    #turn increments into path
    #print(incr)
    copy=[first]
    
    if l_out > 0:     
        for i in range(1,l_out+1):
            copy.append(np.array(copy[i-1]) + np.array(incr[i-1]))

    

    copy=np.array(copy)  
    
    copy=list(copy)
    
    copy=[list(x) for x in copy]
    
    

    return copy


def linsig(position_vector, degree, start=None, end=None):
    """
    Function computes the truncated signature up to degree 'degree' of a single linear path from 0 to a position vector specified.

    Args:
        position_vector (list): a list of coordinates
        degree (int): the highest degree of a signature term to return
        start (float): optional, can input the intial time value, default 0
        stop (float): optional can input the final time value, default 1

    Returns:
        signature (np.array): vector of signature values


    """
    if start==None:
        start=0
    if end==None:
        end=1
    dimension = len(position_vector)
    keys = ts.sigkeys(dimension, degree)
    pos = keys.split()
    pos_out = []
    pos[0]='(0)'
    for row in pos:
        row = row.strip('(')
        row = row.strip(')')
        row = row.split(',')
        
        row=[int(i) for i in row]
        pos_out.append(row)
    pos_out.pop(0)
    signature=[1]
    for i in pos_out:
        sigj=1
        l=len(i)
        for j in i:
            sigj=sigj*position_vector[j-1]
        sigj=sigj*(end-start)**l
        sigj=sigj/math.factorial(l)
        signature.append(sigj)    

    signature=np.array(signature)
    return signature




def isblob(string):
    open=0
    closed =0
    for i in string:
        if i in ['(','[']:
            open=open+1
        if i in [')',']']:
            closed=closed+1
        if closed > open:
            raise SyntaxError('Not a mathematical expression!')
    if open==closed:
        t=1
    else: t=0
    return t


#split_input = input_str.strip().split(' ')

def unbracket(Lie_str):
   # Take a string starting with '[' and ending with ']' and develop it as a Lie bracket, around the central ',' character
    workon=Lie_str[1:-1]
    l=len(workon)
    for i in range(l):
        if workon[i]==',':
            if isblob(workon[:i])==0:
                 pass
            else:
                blob1=workon[:i]
                blob2=workon[i+1:]
    outstring='('+blob1+'x'+blob2+'-'+blob2+'x'+blob1+')'
            
    
   
    return outstring 


# Search for Lie brackets in a string
def spotbracket(string):
    
    l= len(string)
    firstbracket=string.find('[')
    if firstbracket==-1:
        return -1
    else:
        open=1
        closed=0
        i=firstbracket+1
        while open>closed:
        
            if string[i]=='[':
                open=open+1
            if string[i]==']':
                closed=closed+1
            i=i+1
    
        bracket = string[firstbracket:i]


    return bracket

def fullunbracket(lie_string):
    while spotbracket(lie_string)!=-1:
        a=spotbracket(lie_string)
        b=unbracket(a)
        lie_string=lie_string.replace(a,b)
        #index_a=lie_string.find(a)
        #l=len(a)
        #part1=lie_string[:index_a]
        #part2=lie_string[index_a+l:]
        #lie_string=part1+b+part2
    return lie_string

def spotroundbracket(string):
    l= len(string)
    firstbracket=string.find('(')
    if firstbracket==-1:
        return -1
    else:
        open=1
        closed=0
        i=firstbracket+1
        while open>closed:
        
            if string[i]=='(':
                open=open+1
            if string[i]==')':
                closed=closed+1
            i=i+1
    
        bracket = string[firstbracket:i]
    return bracket

def isterm(string):
    t=0
    if isblob(string)==1 and string.replace('-','') != '':
        t=1
    return t

def spotfirstterm(string,j=None):
    if j==None:
        j=0
    string=string[j:]
    l=len(string)
    for i in range(l):

        if string[i]=='-':
            if isterm(string[:i])==1:
                firstterm=string[:i]
                if firstterm.find('(')==-1:
                    return spotfirstterm(string,j=i)
                else:
                    return firstterm
    else:
        return string

def distribute(term):
    """ MUST input a valid TERM"""

    factor=spotroundbracket(term)
    if factor==-1:
        return term
    else:
        factorstring=factor[1:-1]
        #factorposition=factor[1]
        lenfac=len(factorstring)
        middle=int(math.floor(lenfac/2))
        if factorstring[middle] != '-':
            raise SyntaxError ('Not a valid Lie bracket')
        else:
            distrib = term.replace(factor,factorstring[:middle],1)+'-'+term.replace(factor,factorstring[middle+1:],1)
            return distrib

def Lietotensor(Lie_bracket):
    """
    Inputs a nested Lie bracket, returns it as a sum of the basis elements in the tensor algebra.
    """
    tensor_product=fullunbracket(Lie_bracket)
    if len(tensor_product)==1:
        return [(1, [int(tensor_product)])]
    else:
        copy= tensor_product[1:-1]
    while copy.find('(') !=-1:
        term=spotfirstterm(copy)
        d=distribute(term)
        copy=copy.replace(term,d,1)
    string=copy
    string=string.replace('x',',')
    while string.find('--') != -1:
        string=string.replace('--','+')
    while string.find('++')!=-1:
        string=string.replace('++','+')
    while string.find('+-')!=-1:
        string=string.replace('+-','-')
    string=string.replace('+',' +')
    string=string.replace('-',' -')
    string='+'+string
    string=string.split(' ')
    l=len(string)
    for i in range(l):
        j=string[i]
        s=[]
        for k in j[1:]:
            if k!=',':
                s.append(int(k))
        if j[0]=='+':
            j=(1,s)
        else:
            j=(-1,s)
        string[i]=j
    return string   

def changeofbasis(dimension, degree):
    liebasis=ts.logsigkeys(dimension,degree)
    tensorbasis=ts.sigkeys(dimension,degree)
    n=ts.logsigdim(dimension,degree)
    m=ts.sigdim(dimension, degree)
    matrix=np.zeros((n,m))
    pos = tensorbasis.split()
    columnindex = []
    pos[0]='(0)'
    for row in pos:
        row = row.strip('(')
        row = row.strip(')')
        row = row.split(',')
        
        row=[int(i) for i in row]
        columnindex.append(row)
    #print(columnindex)

    rowindex=liebasis.split()
    expand=[]
    for i in range(n):
        B=Lietotensor(rowindex[i])
        #print(B)
        for j in B:
            for k in range(m):
                #print(type(j[1]))
                #print(type(k))
                if j[1]==columnindex[k]:
                    matrix[i][k]=matrix[i][k]+j[0]
    return matrix

def Logsigastensor(array, degree):
    logsig=ts.stream2logsig(array, degree)
    dimension=len(array[0])
    M=changeofbasis(dimension,degree)
    result=np.dot(logsig,M)

    return result

def tensoraslevels(tensor, dimension, degree):
    
    sum = [np.zeros(dimension ** m) for m in range(1,degree + 1)]
    
    counter=0
    for i in range(1, degree+1):
        
        for j in range(dimension**i):
            counter=counter+1
            sum[i-1][j]=tensor[counter]
    return sum


#inputs are values in the tensor algebra given as lists of levels (from 1 to
#level), assumed 0 in level 0.
#returns their concatenation product
def multiplyTensor(a,b):
    level = len(a)
    dim = len(a[0])
    sum = [np.zeros(dim ** m) for m in range(1,level + 1)]
    for leftLevel in range(1,level):
        for rightLevel in range(1, 1 + level - leftLevel):
            sum[leftLevel + rightLevel - 1]+=np.outer(a[leftLevel - 1],b[rightLevel - 1]).flatten()
    return sum

#input is a value in the tensor algebra given as lists of levels (from 1 to
#level), assumed 0 in level 0.
#returns its exp - assumed 1 in level 0
#exp(x)-1 = x+x^2/2 +x^3/6 +x^4/24 + ...
def exponentiateTensor(a):
    out = [i.copy() for i in a]
    level = len(a)
    products = [out]
    for m in range(2,level + 1):
        t = multiplyTensor(a,products[-1])
        for j in t:
            j *= (1.0 / m)
        products.append(t)
    return np.concatenate([np.sum([p[i] for p in products],0) for i in range(level)])
    

def chen(a,b):
    level = len(a)
    dim = len(a[0])
    
    sum = [a[m] + b[m] for m in range(level)]
    for leftLevel in range(1,level):
        for rightLevel in range(1, 1 + level - leftLevel):
            sum[leftLevel + rightLevel - 1]+=np.outer(a[leftLevel - 1],b[rightLevel - 1]).flatten()
            
    return sum

def concatenate(a,b):
    la=len(a)
    dim=len(a[0])
    lb=len(b)
    aa=np.array(a)
    bb=np.array(b)
    bb=bb+aa[(la-1):la]-bb[0:1]
    #print(aa[la-1],aa[(la-1):la])
    return np.vstack([aa,bb[1:]])
   

def paths_generate(dataset_lengths, a, dims):
    """ Generate all possible paths for a chosen number of dimensions, chosen lengths of path steps and a list of allowed modifiers from one step to another
    
    Modifiers of each dimension value are selected randomly from a provided list.

    Args:
        dataset_lengths (list): path lengths for which all possible path combinations are obtained
        a (list): list of modifiers applied at each step of all paths to be obtained. A longer list means more possible paths are obtained
        dims (int): number of dimensions each path step has. Higher dimensionality means more possible paths are obtained.
     
    Returns:
        list: A list of all path combinations given the arguments provided

    Example:
        >>> print (paths_generate([3],[0,1],1))
        [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [1.0]], [[0.0], [1.0], [1.0]], [[0.0], [1.0], [2.0]]]
    """
    # collecting all possible increment combinations for chosen path lengths and the range of chosen changes
    out_increments = []
    for length in dataset_lengths:
        possible_changes = list(itertools.product(a, repeat=dims)) # NUMBER OF DIMENSIONS FOR WHICH PATHS GET GENERATED CAN BE MODIFIED HERE
        temp = list(itertools.product(possible_changes,repeat=length-1))
        out_increments.extend(temp)

    # calculating paths from increments with the starting point 0
    out_paths = []
    for out in out_increments:
        out_path = [list(np.zeros(dims))]
        temp = list(np.zeros(dims))
        for element in out:
            temp = [x + y for x, y in zip(temp, element)]
            out_path.append(temp)
        out_paths.append(out_path)
    return out_paths


def random_path (l,a,dims):
    """ Generate a random path which has a chosen number of steps, given a chosen set of possible modifications and step dimensions


    Args:
        l (int): length of the random path to be produced
        a (list): list of modifiers applied at each step of all paths to be obtained
        dims (int): number of dimensions each path step has

    Returns:
        list: a random path that satisfies the constraints put into the function
    
    Example:
        >>> print (random_path(10,[1,0,-1],2))
        [[-1.0, 0.0], [-1.0, 0.0], [-2.0, 1.0], [-1.0, 2.0], [-2.0, 3.0], [-3.0, 3.0], [-2.0, 2.0], [-2.0, 3.0], [-2.0, 3.0], [-3.0, 4.0]]
    """

    x = list(itertools.product(a, repeat=dims))
    out_increments = []
    for ll in range(l):
        c = random.sample(x, 1)
        out_increments.append(c)
    out_path = []
    temp = list(np.zeros(dims))
    for element in out_increments:
        temp = [x + y for x, y in zip(temp, list(element[0]))]
        out_path.append(temp)
    return out_path



def shuffles (a_list, b_list):
    """ Generates all possible shuffles of two lists
    
    Shuffles as in when shuffling two decks of cards. The resulting shuffles only contain items in the order followed in the 2 lists put into the function. 
    E.g. lists [d,e,f] and [a,b,c] would get shuffled to, among others, form a list [a,d,b,e,c,f] but never a list [b,a,d,c,e,f]

    Args:
        a_list (list): a list of elements
        b_list (list): a list of elements

    Returns:
        list: all possible shuffles of the 2 inputted lists
    
    Example:
        >>> print (shuffles([4],[1,2,3]))
        [[4, 1, 2, 3], [1, 4, 2, 3], [1, 2, 4, 3], [1, 2, 3, 4]]

    To do:
        * this function is very inefficient for generating higher numbers of shuffles, e.g. where both lists have 10 items each

    """
    a = a_list # index values of the first list of numbers
    a_codes = []
    for i, x in enumerate(a):
        x = "a" + str(i)
        a_codes.append(x)
    b = b_list
    b_codes = [] # index values of the first list of numbers
    for i, x in enumerate(b):
        x = "b" + str(i)
        b_codes.append(x)
    c = a_codes+b_codes
    all_mutants = list(itertools.permutations(c, len(c))) # generate all possible combinations of elements from the two lists of numbers
    good_mutants = []
    for x in all_mutants: # pick out combinations suitable for inclusion in shuffled product and collect those in good_mutants list
        temp1 = []
        for v in x: # extract a_codes
            if v[0] == 'a':
                temp1.append(v)
        if temp1 == a_codes: # if order of a_code indices is as in the original, carry on
            pass
        else:
            continue

        temp2 = []
        for v in x: # extract b_codes
            if v[0] == 'b':
                temp2.append(v)
        if temp2 == b_codes: # if order of b_code indices is as in the original, add x to good_mutants
            good_mutants.append(x)

    shuffled = []
    for x in good_mutants:
        x_vals = []
        for v in x:
            if v[0] =='a':
                a_id = v[1:]
                x_vals.append(a[int(a_id)])
            else:
                b_id = v[1:]
                x_vals.append(b[int(b_id)])
        shuffled.append(x_vals)

    return shuffled

