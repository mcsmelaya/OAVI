## Лабораторная работа №1. Цветовые модели и передискретизация изображений

1. Strawberry

- Сохранение отдельных каналов

| ![strawberry.png](src_img/strawberry.png) | ![red_channel_strawberry.png](new_img/red_channel_strawberry.png)  |   ![green_channel_strawberry.png](new_img/green_channel_strawberry.png)    | ![blue_channel_strawberry.png](new_img/blue_channel_strawberry.png)  |
|:-----------------------------------------:|:------------------------------------------------------------------:|:-----------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|
|                 original                  |                            red_channel                             |                                 green_channel                                 |                              blue_channel                               |


- Яркостная компонента, инвертированная яркостная компонента и конвертация из RGB в HSI

| ![intensity_strawberry.png](new_img/intensity_strawberry.png) | ![inverted_intensity_strawberry.png](new_img/inverted_intensity_strawberry.png) | ![rgb_to_hsi_strawberry.png](new_img/rgb_to_hsi_strawberry.png) |
|:-------------------------------------------------------------:|:-------------------------------------------------------------------------------:|:---------------------------------------------------------------:|
|                      intensity                                |                           inverted_intensity                                    |                           rgb_to_hsi                            |

- Интерполяция в M раз, децимация в N раз, передискретизация изображения в K=M/N раз, передискретизация изображения в K раз за один проход

| ![strawberry.png](src_img/strawberry.png) | ![resized_in_2_times_strawberry.png](new_img/resized_in_2_times_strawberry.png) | ![resized_in_3_times_strawberry.png](new_img/resized_in_3_times_strawberry.png) | ![two_pass_resampling_in_23_times_strawberry.png](new_img/two_pass_resampling_in_23_times_strawberry.png) | ![one_pass_resampling_in_1.5_times_strawberry.png](new_img/one_pass_resampling_in_1.5_times_strawberry.png) |
|:-----------------------------------------:|:-------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|
|                 original                  |                              bilinear_resized_M=2                               |                              bilinear_resized_N=3                               |                                               resized_M/N                                                 |                                                resized_K=1.5                                                |


2. Tiger

- Сохранение отдельных каналов

| ![tiger.png](src_img/tiger.png) | ![red_channel_tiger.png](new_img/red_channel_tiger.png) |       ![green_channel_tiger.png](new_img/green_channel_tiger.png)       |     ![blue_channel_tiger.png](new_img/blue_channel_tiger.png)      |
|:------------------------------------:|:-------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------------:|
|               original               |                       red_channel                       |                           green_channel                            |                            blue_channel                            |


- Яркостная компонента, инвертированная яркостная компонента и конвертация из RGB в HSI

| ![intensity_tiger.png](new_img/intensity_tiger.png)  | ![inverted_intensity_tiger.png](new_img/inverted_intensity_tiger.png) | ![rgb_to_hsi_tiger.png](new_img/rgb_to_hsi_tiger.png)  |
|:----------------------------------------------------:|:---------------------------------------------------------------------:|:------------------------------------------------------:|
|                      intensity                       |            inverted_intensity                                         |                       rgb_to_hsi                       |

- Интерполяция в M раз, децимация в N раз, передискретизация изображения в K=M/N раз, передискретизация изображения в K раз за один проход

| ![tiger.png](src_img/tiger.png)  | ![resized_in_2_times_tiger.png](new_img/resized_in_2_times_tiger.png) | ![resized_in_3_times_tiger.png](new_img/resized_in_3_times_tiger.png) | ![two_pass_resampling_in_23_times_tiger.png](new_img/two_pass_resampling_in_23_times_tiger.png)  | ![one_pass_resampling_in_1.5_times_tiger.png](new_img/one_pass_resampling_in_1.5_times_tiger.png) |
|:--------------------------------:|:---------------------------------------------------------------------:|:---------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
|             original             |                         bilinear_resized_M=2                          |                         bilinear_resized_N=3                          |                                           resized_M/N                                            |                                           resized_K=1.5                                           |
