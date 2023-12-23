import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.stats import qmc
import yfinance as yf
from datetime import datetime
from multiprocessing import Pool


# ticker = ['QQQ', '2023-10-25', '2023-11-25', '2m'] #1414 data points
ticker = ['QQQ', '2022-03-1', '2022-09-01', '1h'] #448 data points
print(ticker)

prices_raw = np.array(yf.Ticker(ticker[0]).history(start=ticker[1], end=ticker[2], interval=ticker[3])['Close'])
prices_raw = prices_raw[~np.isnan(prices_raw)]

prices = np.log2(prices_raw / prices_raw[0])

nyquist_freq = (len(prices) // 2) + 1
spectrum = np.fft.fft(prices)[:nyquist_freq]
amplitudes = np.abs(spectrum)


def sim_once(i):
    phases_sim = np.random.uniform(0, 2*pi, size=nyquist_freq)
    spectrum_sim = amplitudes * np.exp(1j * (phases_sim))
    prices_sim = np.fft.ifft(spectrum_sim).real

    if(len(prices)%2==0):
        spectrum_sim = np.concatenate((spectrum_sim[:-1], np.conj(spectrum_sim[-1:0:-1]))) #second half is the negative frequencies, reverse and remove last element, then take last element
    else:
        spectrum_sim = np.concatenate((spectrum_sim, np.conj(spectrum_sim[-1:0:-1])))
    prices_sim = np.fft.ifft(spectrum_sim).real

    # only the first half of the IFFT is reasonable
    halflen = len(prices_sim)//2
    prices_sim = prices_sim[:halflen]

    prices_sim = (prices_sim - prices_sim[0]) #make it start at 0

    x = np.linspace(1e-6, pi/2, len(prices_sim))
    prices_sim = np.multiply(prices_sim, x/np.sin(x)) #scale a y=sin(x) into y=x 

    prices_sim = prices_raw[-1]*np.exp2(prices_sim) #make exponential again and scale by last real price
    return np.int32(prices_sim*100)



if __name__ == '__main__':
    # num_sims = 500_000 # 1mo at 2m interval, 2.83GB
    num_sims = 1_000_000 # 6mo at 1h interval, 1.79GB
    all_sims = np.zeros((num_sims, len(prices_raw)//2), dtype=np.int32)
    with Pool(processes=16) as pool:  # Use as many processes as needed
        result = pool.map(sim_once, range(num_sims))

    all_sims[:] = result 
    print(all_sims.shape)
    # np.savetxt('output.csv', np.round(all_sims,2), delimiter=',')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = '_'.join(ticker) + '_' + str(num_sims) + 'n_' + timestamp
    np.save(f'./opt_saved/{file_name}.npy', all_sims)




