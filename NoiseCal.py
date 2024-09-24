import mpmath as mp
import pandas as pd
import copy
mp.mp.dps = 500;

def calculate_sigma_acc_lmkcdey(n, q, N, sigma, d_g, B_g, t, delta, w, Xs='ternary'):
    norm_s_N_square = 0
    
    if(Xs == "ternary"):
        norm_s_N_square = N/2
    elif(Xs == "binary"):
        norm_s_N_square = N/2
    else:
        print("hello")
        #sqrt(n(N)*sigma^2)

    # worst-case
    # k = n

    # average-case
    k = N * (1 - mp.exp(-1 * n / N ))

    cutoff_br_term = 1
    approx_gadget_decomp_term = 0

    if(t != 0):     # Using cutoff blind rotation
        cutoff_br_term = (1 - (2*t+1)/q)

    if(delta != 1):
        approx_gadget_decomp_term = mp.power(delta, 2) / 12 * (norm_s_N_square + 1)

    rlwep_term = (d_g * N * mp.power(B_g, 2) / 12)
    lmk_term = (k + (N - k) / w)

    sigma_acc_lmkcdey = (rlwep_term + approx_gadget_decomp_term) * ( 2 * n * mp.power(sigma, 2) * cutoff_br_term + lmk_term * mp.power(sigma, 2)) 
    return sigma_acc_lmkcdey

def calculate_sigma_acc_ap(n, q, N, sigma, d_r, d_g, B_g, t, delta, Xs='ternary'):
    norm_s_N_square = 0
    
    if(Xs == "ternary"):
        norm_s_N_square = N/2
    elif(Xs == "binary"):
        norm_s_N_square = N/2
    else:
        print("hello")
        #sqrt(n(N)*sigma^2)

    cutoff_br_term = 1
    approx_gadget_decomp_term = 0

    if(t != 0):     # Using cutoff blind rotation
        cutoff_br_term = (1 - (2*t+1)/q)
    if(delta != 1): # Using approximated gadget decomposition
        approx_gadget_decomp_term = mp.power(delta, 2) / 12 * (norm_s_N_square + 1)

    sigma_acc_ap = n * d_r * 2 * ( d_g * N * mp.power(B_g, 2) / 12 * mp.power(sigma, 2) + approx_gadget_decomp_term) * cutoff_br_term

    return sigma_acc_ap

def calculate_sigma_acc_ginx(n, q, N, sigma, d_g, B_g, t, delta, Xs='ternary'):
    u = 0
    if(Xs == "ternary"):
        u = 2
        norm_s_N_square = N/2
    elif(Xs == "binary"):
        u = 1
        norm_s_N_square = N/2
    else: # Gaussian
        print('hello')
        #sqrt(n(N)*sigma^2) 

    cutoff_br_term = 1
    approx_gadget_decomp_term = 0

    if(t != 0):     # Using cutoff blind rotation
        cutoff_br_term = (1 - (2*t+1)/q)
    if(delta != 1): # Using approximated gadget decomposition
        approx_gadget_decomp_term = mp.power(delta, 2) / 12 * (norm_s_N_square + 1)
    
        
    sigma_acc_ginx = 2 * u * 2 * n * ( d_g * N * mp.power(B_g, 2) / 12 * mp.power(sigma, 2) + approx_gadget_decomp_term) * cutoff_br_term
    return sigma_acc_ginx

def calculate_sigma_else(n, N, sigma, d_ks, Xs='ternary'):
    norm_s_N_square = 0
    norm_s_n_square = 0
    
    if(Xs == "ternary"):
        norm_s_N_square = N/2
        norm_s_n_square = n/2
    elif(Xs == "binary"):
        norm_s_N_square = N/2
        norm_s_n_square = n/2
    else:
        print("hello")
        #sqrt(n(N)*sigma^2)
        
    sigma_ms1 = (norm_s_N_square + 1) / 3
    sigma_ms2 = (norm_s_n_square + 1) / 3    
    sigma_ks = mp.power(sigma, 2) * N * d_ks
    
    return sigma_ms1, sigma_ms2, sigma_ks

def calculate_total_stddev(parameters, Xs, method = 'AP'):
    sigma = parameters['sigma']
    n = parameters['n']
    q = parameters['q']
    N = parameters['N']
    d_r = parameters['d_r']
    d_g = parameters['d_g']
    d_ks = parameters['d_ks']
    B_g = parameters['B_g']
    t = parameters['t']
    Q_ks = parameters['Q_ks']
    Q = parameters['Q']
    delta = parameters['delta']
    w = parameters['w']
    
    sigma_acc = 0
    if(method == 'AP'):
        sigma_acc = calculate_sigma_acc_ap(n, q, N, sigma, d_r, d_g, B_g, t, delta, Xs)
    elif(method == 'GINX'):
        sigma_acc = calculate_sigma_acc_ginx(n, q, N, sigma, d_g, B_g, t, delta, Xs)
    elif(method == 'LMKCDEY'):
        sigma_acc = calculate_sigma_acc_lmkcdey(n, q, N, sigma, d_g, B_g, t, delta, w, Xs)

    threshold_error = 0
    if(t != 0): # Using cutoff blindrotation
        threshold_error = 2 * n * mp.power(t, 3) / (3 * q)

    sigma_ms1, sigma_ms2, sigma_ks = calculate_sigma_else(n, N, sigma, d_ks, Xs)
    sigma_total = mp.power(q, 2) / mp.power(Q_ks, 2) * ((mp.power(Q_ks, 2) / mp.power(Q, 2)) * 2 * sigma_acc + sigma_ms1 + sigma_ks) + sigma_ms2
    stddev_total = mp.sqrt(sigma_total + threshold_error)

    return stddev_total 

def calculate_failure_porb(parameters, Xs, method = 'AP'):
    sigma = parameters['sigma']
    n = parameters['n']
    q = parameters['q']
    N = parameters['N']
    d_r = parameters['d_r']
    d_g = parameters['d_g']
    d_ks = parameters['d_ks']
    B_g = parameters['B_g']
    t = parameters['t']
    Q_ks = parameters['Q_ks']
    Q = parameters['Q']
    delta = parameters['delta']
    
    stddev_total = calculate_total_stddev(parameters, Xs, method)
    result = mp.log(1 - mp.erf((q/8) / (mp.sqrt(2)*stddev_total)), 2)

    return round(result, 10)

def computation_complex(parameters):
    # if threshold is 0, it will be original complexity
    n = parameters['n']
    q = parameters['q']
    d_r = parameters['d_r']
    d_g = parameters['d_g']
    B_r = parameters['B_r']
    t = parameters['t']
    proposed_complexity = 2 * (1 - 1/B_r) * n * (1 - (2*t+1)/q) * d_r * (d_g + 1)
    return round(proposed_complexity, 10)

def number_of_mult_ap(parameters):
    # if threshold is 0, it will be original complexity
    n = parameters['n']
    q = parameters['q']
    d_r = parameters['d_r']
    d_g = parameters['d_g']
    B_r = parameters['B_r']
    t = parameters['t']
    proposed_complexity = 2 * (1 - 1/B_r) * n * (1 - (2*t+1)/q) * d_r
    return round(proposed_complexity, 10)

def number_of_mult_ginx(parameters, Xs='ternary'):
    # if threshold is 0, it will be original complexity
    n = parameters['n']
    q = parameters['q']
    t = parameters['t']
    U = 1
    if Xs=='ternary':
        U = 2
    proposed_complexity = 2 * U * n * (1 - (2 * t + 1) / q)
    return round(proposed_complexity, 10)

def number_of_mult_lmkcdey(parameters, Xs='ternary'):
    # if threshold is 0, it will be original complexity
    n = parameters['n']
    N = parameters['N']
    q = parameters['q']
    t = parameters['t']
    w = parameters['w']

    # worst-case
    # k = n
    # average-case
    k = N * (1 - mp.exp(-1 * n / N ))

    proposed_complexity = 2 * n * (1 - (2 * t + 1) / q) + (w-1) / w * k + N / w + 2
    return round(proposed_complexity, 10)

def btkSize(parameters):
    n = parameters['n']
    N = parameters['N']
    Q = parameters['Q']
    d_r = parameters['d_r']
    d_g = parameters['d_g']
    B_r = parameters['B_r']
    t = parameters['t']
    btksize = 4 * n * N * d_r * (B_r - 1) * d_g * mp.log(Q, 2)
    return round(btksize, 10)

def display_parameters(parameters):
    dummyParam = copy.deepcopy(parameters)
    dummyParam['FP_AP']   = calculate_failure_porb(dummyParam, Xs = 'ternary', method = 'AP')
    dummyParam['FP_GINX'] = calculate_failure_porb(dummyParam, Xs = 'ternary', method = 'GINX')
    dummyParam['MEM']     = btkSize(dummyParam)
    
    # Q 제거
    if 'Q' in dummyParam:
        del dummyParam['Q']
    # Q_ks 제거
    if 'Q_ks' in dummyParam:
        del dummyParam['Q_ks']
        
    #B_g, B_ks, B_r 2^n 꼴로 변경
    dummyParam['raw B_g'] = dummyParam['B_g']
    dummyParam['B_g'] = mp.ceil(mp.log(dummyParam['B_g'], 2))
    dummyParam['B_ks'] = mp.ceil(mp.log(dummyParam['B_ks'], 2))
    dummyParam['B_r'] = mp.ceil(mp.log(dummyParam['B_r'], 2))
    
    df = pd.DataFrame.from_dict(dummyParam, orient='index', columns=['Value'])
    df.index.name = 'Parameter'
    df.columns.name = ''
    return df.transpose()

def display_parameters_vector(*parameters_vec, Xs = 'ternary'):
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    dfvec = []
    index = 0
    for param in parameters_vec:
        dummyParam = copy.deepcopy(param)
        dummyParam['index'] = index
        dummyParam['FP_AP']   = calculate_failure_porb(dummyParam, Xs)
        dummyParam['FP_GINX'] = calculate_failure_porb(dummyParam, Xs = Xs, method = 'GINX')
        dummyParam['FP_LMKCDEY'] = calculate_failure_porb(dummyParam, Xs = Xs, method = 'LMKCDEY')
        # dummyParam['MEM'] = btkSize(dummyParam)
        # dummyParam['CC'] = computation_complex(dummyParam)
        dummyParam['# of RLWE\' AP'] = number_of_mult_ap(dummyParam)
        dummyParam['# of RLWE\' GINX'] = number_of_mult_ginx(dummyParam)
        dummyParam['# of RLWE\' LMKCDEY'] = number_of_mult_lmkcdey(dummyParam)
        
        # Q 제거
        if 'Q' in dummyParam:
            del dummyParam['Q']
        # Q_ks 제거
        if 'Q_ks' in dummyParam:
            del dummyParam['Q_ks']
        
        #B_g, B_ks, B_r 2^n 꼴로 변경
        dummyParam['raw B_g'] = dummyParam['B_g']
        dummyParam['B_g'] = mp.ceil(mp.log(dummyParam['B_g'], 2))
        dummyParam['B_ks'] = mp.ceil(mp.log(dummyParam['B_ks'], 2))
        dummyParam['B_r'] = mp.ceil(mp.log(dummyParam['B_r'], 2))
        dummyParam['w'] = dummyParam['w']

        
        df = pd.DataFrame.from_dict(dummyParam, orient='index', columns=[dummyParam['name']])
        df.index.name = 'Parameter'
        df.columns.name = ''
        dfvec.append(df)
        index += 1
    
    df_combined = pd.concat(dfvec, axis=1)
    return df_combined.transpose()

def parse_data(data):
    lines = data.strip().split('\n')
    param_data = lines[1:]

    result = []
    
    for line in param_data:
        line = line.strip().rstrip(' }, ').lstrip('{ ').replace(' ', '').split(',')
        name = line[0]
        params = line[1:]
        
        if(params[4] == 'PRIME'):
            params[4] = 1
        
        param_dict = {
            'name': name,
            'n': int(params[2]),
            'q': int(params[3]),
            'N': int(params[1]) / 2,
            'logQ':    int(params[0]),
            'logQ_ks': mp.log(int(params[4]),2),
            'Q': mp.power(2, int(params[0])),
            'Q_ks': int(params[4]),
            'B_ks': int(params[6]),
            'B_g':  int(params[7]),
            'B_r':  int(params[8]),
            'w': int(params[9]),
            't': 0,
            'delta': 1,
            'sigma': mp.mpf('3.2'),  # Assuming a standard deviation
            'd_r':  mp.ceil(mp.log(int(params[3]), int(params[8]))),  # log_{B_r}^{q}
            'd_g':  mp.ceil(mp.log(mp.power(2, int(params[0])), int(params[7]))),  # log_{B_g}^{Q}
            'd_ks': mp.ceil(mp.log(mp.power(2, mp.log(int(params[4]),2)), int(params[6])))  # log_{B_ks}^{Q_ks}
        }
        
        result.append(param_dict)

    return result

def setDelta(data, Delta):
    data['delta'] = mp.power(2, Delta)
    logQ = data['logQ']
    log_Bg = mp.log(data['B_g'], 2)
    data['d_g']    = mp.ceil((logQ - Delta)/log_Bg)

def setCutoff(data, t):
    data['t'] = t

def setBaseG(data, BaseG):
    data['B_g'] = mp.power(2, BaseG)
    logQ = data['logQ']
    Delta = mp.log(data['delta'], 2)
    data['d_g']    = mp.ceil((logQ - Delta)/BaseG)


# Caution.
# It changes 3 variables : DigitsG, BaseG, Delta
# It will be find optimum BaseG and Delta that makes minimum failure probability
def setDigitsG(data, DigitsG, method = 'AP'):
    MinFP = 1
    data['d_g'] = DigitsG
    logQ = data['logQ']
    maxBg = int(mp.ceil(logQ/DigitsG))

    # Bg in [ 2^0, ceil(logQ/d_g)]
    for Bg in range(1, maxBg + 1):
        tmp = copy.deepcopy(data)
        # Delta = logQ - DigitsG * BaseG
        Delta = mp.power(2, logQ - DigitsG * Bg)
        tmp['delta'] = Delta
        tmp['B_g'] = mp.power(2, Bg)
        FP = calculate_failure_porb(tmp, 'ternary', method)
        if(MinFP > FP):
            MinFP = FP
            data['delta'] = Delta
            data['B_g']   = mp.power(2, Bg)
    
            


## Data 입력 함수도 만들어두기.