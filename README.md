Проект по нахождению оптимального управления процессом нагрева куба.

Данный проект решает задачу:
![image](https://github.com/YuliaMerkulova/Optimal-control/assets/31349056/9ecb62d9-8ef5-4741-9ee4-250e99b6405d)
где f - распределение температуры, а u - управление.

Для решения задачи используется метод проекции градиента. Программа ускорена с помощью библиотеки Numba на CPU, графики строятся с помощью matplotlib в виде проекций функций на ось z = 0.

Результат работы алгоритма при l = 20, T = 20, h = 1, tau = 1:
![image](https://github.com/YuliaMerkulova/Optimal-control/assets/31349056/52b7ba48-71a8-49a3-8ff5-39bee2d5be12)

