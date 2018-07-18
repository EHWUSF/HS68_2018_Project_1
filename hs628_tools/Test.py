
import numpy as np

def test(dataset):
    dataset = np.genfromtxt('heart.csv', delimiter =',', skip_header = 1)

    for i in range(len(dataset)):
        if dataset[i][:] == 'NA':
            print 'what'

test(dataset)