import allantools
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    # t = np.logspace(-4, 3, 2000)  # tau values 
    y = -1 + 2 * np.random.normal(loc=0, scale=1, size=100*60*20)  # 20 minutes of gaussian white noise with 1-sigma of 1.0
    r = 800  # sample rate in Hz
    (t2, ad, ade, adn) = allantools.oadev(y, rate=r, data_type="freq", taus="all")  # Compute the overlapping ADEV
    fig = plt.loglog(t2, ad)  # Plot the results
    plt.title('Overlapping ADEV Curve')
    plt.xlabel('Averaging Time (seconds)')
    plt.ylabel('Allan Deviation (1-sigma)')
    # plt.show()
    plt.savefig('allan_0.png')

    '''
    The diagonal element in the R covariance matrix is the squared value of the y value corresponding to the
    sampling time.  In this example case, this is seen with the correlation of the value of 1.0 at the time
    1 / 800 seconds, which is the noise profile generated. 
    '''