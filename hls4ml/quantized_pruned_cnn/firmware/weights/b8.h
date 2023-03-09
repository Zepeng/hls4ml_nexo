//Numpy array shape [16]
//Min -0.125000000000
//Max 0.250000000000
//Number of zeros 4

#ifndef B8_H_
#define B8_H_

#ifndef __SYNTHESIS__
bias8_t b8[16];
#else
bias8_t b8[16] = {0.25000, 0.00000, 0.09375, 0.62500, 0.40625, 0.31250, 0.00000, 0.03125, 0.34375, -0.37500, 0.00000, 0.03125, -0.31250, 0.87500, -0.09375, 0.21875};
#endif

#endif
