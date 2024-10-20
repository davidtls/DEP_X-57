
import numpy as np


# Case 1:
alpha_1 = np.array([-2.013574661, 1.923076923, 5.968325792, 7.963800905, 9.932126697, 10.99095023, 11.94117647,
                    12.89140271, 13.95022624, 14.92760181, 15.87782805, 16.9638009, 17.91402715])
CL_1 = np.array([1.546120059, 2.161054173, 2.740849195, 2.972181552, 3.209370425, 3.314787701, 3.420204978, 3.267935578,
                 3.308931186, 3.039531479, 3.045387994, 2.811127379, 2.82284041])
CD_1 = np.array([0.152054795, 0.20239726, 0.279452055, 0.321575342, 0.36369863, 0.383219178, 0.403767123, 0.457191781,
                 0.490068493, 0.535273973, 0.576369863, 0.613356164, 0.655479452])

# Case 2:
alpha_2 = np.array([-2.071307301, 2.003395586, 4.006791171,	8.013582343, 10.01697793, 11.00169779, 12.05432937,
                    12.97113752, 13.98981324, 15.04244482,	15.99320883, 16.97792869, 18.03056027])
CL_2 = np.array([2.156160458, 2.908309456, 3.259312321,	3.868194842, 4.140401146, 4.247851003, 4.333810888, 4.391117479,
                 4.419770774, 4.183381089, 4.011461318, 3.975644699, 3.93982808])

alpha_3 = np.array([-2.055045872, 1.981651376, 4, 8, 10.01834862, 11.00917431, 12.03669725, 13.02752294, 14.01834862,
                    15.08256881, 16.03669725, 16.99082569, 18.01834862])
CD_2 = np.array([0.236645963, 0.346583851, 0.411801242, 0.544099379, 0.61863354, 0.661490683, 0.691304348, 0.719254658,
                 0.743478261, 0.790062112, 0.845962733, 0.885093168, 0.926086957])
alpha_4 = np.array([-1.971830986, 1.971830986, 3.943661972,	7.992957746, 9.964788732, 11.02112676, 12.00704225,
                    15.1056338,	16.02112676, 17.00704225, 18.02816901, 18.02816901])
Cm = np.array([1.084202683,	0.762295082, 0.64828614, 0.192250373, -0.578986587, -0.847242921, -1.21609538, -1.658718331,
               -1.933681073, -2.074515648, -2.22876304, -2.101341282])


DATA1 = np.stack((alpha_1, CL_1, CD_1))
DATA2 = np.stack((alpha_2, CL_2))
DATA3 = np.stack((alpha_3, CD_2))
DATA4 = np.stack((alpha_4, Cm))

