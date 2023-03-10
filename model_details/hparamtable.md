
# Variable decay

| arch. family | depth | batch size | initial lr |   decay |
|---|---|---|---|---|
| Conv Actually | 4 |  1024 | $10^{-4}$  |    `10**linspace(-5, -1, 4)` |
| Myrtle CNN | N/A |  512 | $10^{-2}$  |    `10**linspace(-5, -2, 10)` |
| CIFAR VGG | 11 |  256 | $10^{-2}$  |   `10**linspace(-5, -2, 10)` |
| ImageNette VGG | 11 |  256 | $1.250 \times 10^{-3}$   |   `10**linspace(-5, -2, 3)` |
| ResNet | 9 |  512 | $10^{-1}$  |   `10**linspace(-5, -2, 10)` |

# Variable depth

| arch. family | depth |  batch size | initial lr |   max epochs | 
|---|---|---|---|---|
| Conv Actually | 1,2,4,8 |  1024 | $10^{-4}$  |   500 | 
| CIFAR VGG | 11, 13, 16, 19 |  256 | $10^{-2}$  |   500 |
| ImageNette VGG | 11, 13, 16 , 19 |  256 | $1.250 \times 10^{-3}$   |   500 |
| ResNet | 9, 20, 56 |  512 | $10^{-1}$  |   200 |