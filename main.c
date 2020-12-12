#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <locale.h>

#define iter 4 // Количество итераций по градиентному методу
#define eps 0.0001 // Точность для шага(чтобы не был 0)
#define lambda 0.0
#define STEP 0.2
#define SENDLAY 2 // количество перцептронов на 2ом слое


void generate();

void normalize(float *a, int N){
    float res = 0., res_r;
    int i;
    for(i = 0; i < N; ++i){
        res += a[i] * a[i];
    }
    res_r = 1 / sqrt(res);
    for(i = 0; i < N; ++i){
        a[i] *= res_r;
    }
}

void init(float* a, int N) {
    int i;
    for (i = 0; i < N; ++i) {
        a[i] = eps;
    }
}

void out(float* omega, int N, const char a[]){
    int i,j;
    printf("Weights \n");
    for (i = 0; i < N; ++i) {
        j = 0;
        while(a[j] != '\0'){
            printf("%c", a[j]);
            ++j;
        }
        printf("[%d] = %f,\n", i + 1, omega[i]);
    }

}

void out_omega(float* omega, int N) {
    int i;
    printf("Weights \n");
    for (i = 0; i < N; ++i) {
        printf("omega[%d] = %f,\n", i + 1, omega[i]);
    }
    printf("\n");

}

void out_omega_TMP(float* omega, int N, float acc) {
    int i;
    FILE* in;
    const char name[] = "C:/TMP.txt";
    if ((in = fopen(name, "a")) == NULL) {
        printf("not_open_1");
        return ;
    }
    fprintf(in, "Acc = %.2f%%\n", acc);
    for (i = 0; i < N; ++i) {
        if(i != N-1) fprintf(in, "Omega[%d] = %.4f,\n", i + 1, omega[i]);
        else fprintf(in, "Omega[%d] = %.4f.\n", i + 1, omega[i]);
    }
    fprintf(in, "\n");

}

float grad(float* a, int N, float* omega, int i, char mode) {
    int j;
    float res = 0.;
    float tmp_omega = 0.;
    for (j = 0; j < N; ++j) {
        res += a[j] * omega[j];
    }
    switch (mode){
    case 'l':
        res = 2 * a[i] * (res - a[N]);
        res += lambda * omega[i];
        break;
    case 's':
        res = exp(-res);
        float ex = 1./(1. + res);
        res = 2. * (ex - a[N]) * res * a[i] * ex * ex;
        break;
    case 'd':
        res = 0.;
        for (j = 0; j < N; ++j) {
            res += a[j] * omega[j] * omega[j];
        }
        res = exp(-res);
        ex = 1./(1. + res);
        res = 4. * (ex - a[N]) * res * a[i] * omega[i] * ex * ex;
        break;
    case 'n':
        for(j = 0; j < N; ++j){
            tmp_omega += omega[j] * omega[j];
        }
        res = 2. * (res / (sqrt(tmp_omega)) - a[N]) *
              (a[i] * tmp_omega - omega[i] * res / sqrt(tmp_omega)) / tmp_omega;

        res += lambda * omega[i];
        break;
    default:
        printf("ERROR!");
        break;
    }
    return res;
}

float func(float* b, int N, float* omega, char mode) {
    float res = 0.;
    float tmp_b = 0., tmp_omega = 0.;
    int i;
    for (i = 0; i < N; ++i) {
        res += omega[i] * b[i];
    }
    if(mode == 'd' || mode == 'f'){
        res = 0.;
        for (i = 0; i < N; ++i) {
            res += omega[i] * b[i] * omega[i];
        }
    }
    switch (mode){
    case 'l':
        break;
    case 's':
        res = 1./(1. + exp(-res));
        break;
    case 'd':
        res = 1./(1. + exp(-res));
        break;
    case 'n':
        for(i = 0; i < N; ++i){
            tmp_b += b[i] * b[i];
            tmp_omega += omega[i] * omega[i];
        }
        res = res / (sqrt(tmp_b) * sqrt(tmp_omega));
        break;
    default:
        printf("ERROR!");
        break;
    }
    return res;
}

float fchecker(float a){
    return a > 0. ? 1. : 0.;
}

float grad_t(float* a, int N, float* omega, int i, char mode, int prizn) {
    int j;
    float res = 0.;
    float tmp_omega = 0., tmp_a = 0.;
    for (j = 0; j < N; ++j) {
        res += a[j] * omega[j];
    }
    switch (mode){
    case 'l':
        res = 2 * a[i] * (res - fchecker(a[prizn]));
        res += lambda * omega[i];
        break;
    case 's':
        res = exp(-res);
        float ex = 1./(1. + res);
        res = 2. * (ex - fchecker(a[prizn])) * res * a[i] * ex * ex;
        break;
    case 'd':
        res = 0.;
        for (j = 0; j < N; ++j) {
            res += a[j] * omega[j] * omega[j];
        }
        res = exp(-res);
        ex = 1./(1. + res);
        res = 4. * (ex - fchecker(a[prizn])) * res * a[i] * omega[i] * ex * ex;
        break;
    case 'n':
        for(j = 0; j < N; ++j){
            tmp_omega += omega[j] * omega[j];
            tmp_a += a[j] * a[j];
        }
//        res = 2. * (res / (sqrt(tmp_omega)) - fchecker(a[prizn])) *
//              (a[i] * tmp_omega - omega[i] * res / sqrt(tmp_omega)) / tmp_omega;
        res = 2. * (res / (sqrt(tmp_a) * sqrt(tmp_omega)) - fchecker(a[prizn])) *
              (a[i] * tmp_omega - omega[i] * res / sqrt(tmp_omega)) /
              sqrt(tmp_a) / tmp_omega;
        res += lambda * omega[i];
        break;
    default:
        printf("ERROR!");
        break;
    }
    return res;
}

void train(float *old_mas, float *new_mas, float **a, int N, int k, char mode, int priznak){
    int i, j, l;
    float step = STEP;
    for (i = 0; i < k; ++i) {
        for (l = 0; l < iter; ++l) {
            for (j = 0; j < N; ++j) {
                new_mas[j] = old_mas[j] - step * grad_t(a[i], N, old_mas, j, mode, priznak);
            }
            for (j = 0; j < N; ++j) {
                old_mas[j] = new_mas[j];
            }
            step /= 2.;
            if (step - eps < 0.) {
                step = STEP;
            }
        }
    }

}
// Обучает 2ой слой
// Е
void train_2(float *old_mas, float *new_mas, float **a, float *omega,
             float *sigma, int N, int k, char mode, int priznak){
    int i, j, l;
    float step = STEP;
    float g[3];
    for (i = 0; i < k; ++i) {
        g[0] = func(a[i], N, omega, mode);
        g[1] = func(a[i], N, sigma, mode);
        g[2] = a[i][N];
        for (l = 0; l < iter; ++l) {
            for (j = 0; j < N; ++j) {
                new_mas[j] = old_mas[j] - step * grad_t(g, 2, old_mas, j, mode, 2);
            }
            for (j = 0; j < N; ++j) {
                old_mas[j] = new_mas[j];
            }
            step /= 2.;
            if (step - eps < 0.) {
                step = STEP;
            }
        }
    }

}

char info(){
    char mode;
    printf("---------------------------------------------------------------------------\n");
    printf("l - Линейная комбинация\ns - Сигмоида от линейной комбинации\nd - Сигмоида, которая принимает линейную комбинацию, но омеги в квадрате\n");
    printf("n - Нормированная линейная комбинация\nq - Выход \n");
    printf("---------------------------------------------------------------------------\n");
    scanf("%c", &mode);
    while ((getchar()) != '\n');
    return mode;
}

int main_1();

int main(){
    setlocale(LC_ALL, "Russian");
    char mode;
    printf("Сгенерировать данные? \n y\\n? \n");

    scanf("%c", &mode);
    while ((getchar()) != '\n');
    if(mode == 'y'){
        generate();
    } else if (mode == 'n'){
        main_1();
    }
// TEST

// TEST
    return 0;
}

int main_1()
{
    char mode;
    while(1){
    // Нейронка
        mode = info();
        if(mode == 'q'){
            break;
        }
        int i, j;
        char c;
//        float step = STEP; // Шаг метода
        int N; // Размерность векторов(без учета разметки)
        int k; // Количество векторов обучающейся выборки
// Начало считывания из файла
        FILE* in;
        const char name[] = "C:/input_learn_g.txt";
        if ((in = fopen(name, "r")) == NULL) {
            printf("not_open_1");
            return 0;
        }
        fscanf(in, "%d%d", &k, &N);
//        float a[k][N+1];
        float **a;
        a = malloc(k*sizeof(int));
        for(i = 0; i < k; ++i){
            a[i] = malloc((N+1)*sizeof(float));
        }
        float old_omega[N], new_omega[N];
        float old_sigma[N], new_sigma[N];
        float old_delta[SENDLAY], new_delta[SENDLAY];

        init(old_omega, N);
        init(new_omega, N);
        init(old_sigma, N);
        init(new_sigma, N);
        init(old_delta, SENDLAY);
        init(new_delta, SENDLAY);

        for (i = 0; i < k; ++i) {
            for (j = 0; j < N; ++j) {
                fscanf(in, "%f", &a[i][j]);
                fscanf(in, "%c", &c);
            }
            fscanf(in, "%f", &a[i][N]);
        }
        fclose(in);
// Конец считывания из файла
// Нормализация данных
        for (i = 0; i < k; ++i) {
            normalize(a[i], N);
        }
// Конец нормализации

// Начало обучения
// Обучение 1го слоя
        int tmp_N = N;
        train(old_omega, new_omega, a, N, k, mode, 0);
        train(old_sigma, new_sigma, a, N, k, mode, 1);
//  Обучение 2го слоя
        train_2(old_delta, new_delta, a, new_omega, new_sigma, N, k, mode, N);
        N = tmp_N;

    // Конец обучения
    free(a);
    // Тестирование
        int pogr = 0; // Итоговая ошибка
        float check = 0.;
        float checker = 0.;// Переменная, которая определяет разграничение хорошего и плохого класса
        switch (mode){
        case 'l':
            checker = 0.5;
            break;
        case 's':
            checker = 0.5;
            break;
        case 'd':
            checker = 0.5;
            break;
        case 'f':
            checker = 0.5;
            break;
        case 'n':
            checker = 0.666666666;
           // checker = 0.75; TODO подумай над этим
            break;
        default:
            printf("ERROR!");
        }
        const char name_1[] = "C:/input_test_g.txt";
        if ((in = fopen(name_1, "r")) == NULL) {
            printf("not_open_2");
            return 0;
        }
        fscanf(in, "%d", &k);
        float b[k][N+1];
        for (i = 0; i < k; ++i) {
            for (j = 0; j < N; ++j) {
                fscanf(in, "%f", &b[i][j]);
                fscanf(in, "%c", &c);
            }
            fscanf(in, "%f", &b[i][N]);
        }
        float temp[SENDLAY];
        for (i = 0; i < k; ++i) {
            temp[0] = func(b[i], N, new_omega, mode);
            temp[1] = func(b[i], N, new_sigma, mode);
            if (func(temp, SENDLAY, new_delta, mode) >= checker) {
                check = 1.;
            }
            else {
                check = 0.;
            }
            if (fabs(check - b[i][N]) == 0) {
                ++pogr;
            }
            printf("Func = %f --- Predict = %f \n", func(temp, SENDLAY, new_delta, mode), b[i][N]);
        }



        out(new_omega, N, "omega");
        out(new_sigma, N, "sigma");
        out(new_delta, SENDLAY, "delta");

        printf("Accuracy = %f%%\n", ((float)pogr / (float)k) * 100.);
//        out_omega_TMP(old_omega, N, ((float)pogr / (float)k) * 100.);
        free(b);
    }

    return 0;
}


// Часть кода, которая отвечает за генерацию
// TODO - добавить возможность произвольной генерации
// Функции для генерации


float gauss(void){

    bool flag;
    float u, v, s = 2.;
    label:
        flag = true;
        u = (float)rand() / (float)RAND_MAX * 2. - 1.;
        v = (float)rand() / (float)RAND_MAX * 2. - 1.;
        while(flag){
            if(u * u + v * v - 1. < 0.){
                flag = false;
            } else {
                u = (float)rand() / (float)RAND_MAX * 2. - 1.;
                v = (float)rand() / (float)RAND_MAX * 2. - 1.;
            }
        }
        s = u * u + v * v;
        s = u * sqrt(-2. * logf(s) / s);
        if(fabs(s) > 1.) goto label;
    return s;
}

// Генерит случайное число от -est/2 до est/2
float fgen(float est, char mode) {
    switch (mode){
    case 'g':
        return gauss();
    case 'r':
        return (float)rand() / (float)RAND_MAX * est - est * 0.5;
    default:
        printf("ERROR!!!");
        return -1;
    }
}

float fcheck(float temp_0, float temp_1) {
    if(temp_0 > 0. && temp_1 > 0.){
        return 1.;
    } else {
        return 0.;
    }
}

void generate() {
    int i, j;
    srand(time(NULL));
    FILE* in_t;
    FILE* in_l;
    const char name_t[] = "C:/input_test_g.txt";
    const char name_l[] = "C:/input_learn_g.txt";
    if ((in_t = fopen(name_t, "w")) == NULL) {
        printf("Cant create input_test_g.txt");
    }
    if ((in_l = fopen(name_l, "w")) == NULL) {
        printf("Cant create input_learn_g.txt");
    }
    int count;
    char mode;
    printf("Количество обучающий векторов(тестовых в 3 раза меньше) :");
    scanf("%d", &count);
    printf("---------------------------------------------------------------------------\n");
    printf("g - Нормальное распределение\nr - Равномерное распределение\n");
    printf("---------------------------------------------------------------------------\n");
    char mode_2 = ' ';
    getchar();
    scanf("%c", &mode_2);
    int dem;
    printf("Размерность векторов :");
    scanf("%d", &dem);
    printf("Пределы генерации отличаются? \n y\\n? \n");
    getchar();
    scanf("%c", &mode);
    float rand_n[dem];
    if(mode == 'y'){
        for(i = 0; i < dem; ++i){
            printf("Предел генерации %d компоненты:", i+1);
            scanf("%f", &rand_n[i]);
        }
    } else {
        printf("Пределы генерации:");
        scanf("%f", &rand_n[0]);
        for(i = 1; i < dem; ++i){
            rand_n[i] = rand_n[0];
        }
    }
    fprintf(in_l, "%d %d\n", count, dem);
    fprintf(in_t, "%d \n", count/3);
    float temp_0, temp_1;
    for (i = 0; i < count; ++i) {
        temp_0 = fgen(rand_n[0], mode_2);
        temp_1 = fgen(rand_n[1], mode_2);
        for(j = 0; j < dem; ++j){
            if(j == 1){
                fprintf(in_l, "%f ", temp_1);
            } else if(j == 0){
                fprintf(in_l, "%f ", temp_0);
            } else {
                fprintf(in_l, "%f ", fgen(rand_n[j], mode_2));
            }
        }
        fprintf(in_l, "%f\n", fcheck(temp_0, temp_1));
    }
    for (i = 0; i < count/3; ++i) {
        temp_0 = fgen(rand_n[0], mode_2);
        temp_1 = fgen(rand_n[1], mode_2);
        for(j = 0; j < dem; ++j){
            if(j == 1){
                fprintf(in_t, "%f ", temp_1);
            } else if(j == 0){
                fprintf(in_t, "%f ", temp_0);
            } else {
                fprintf(in_t, "%f ", fgen(rand_n[j], mode_2));
            }
        }
        fprintf(in_t, "%f\n", fcheck(temp_0, temp_1));
    }
    fclose(in_l);
    fclose(in_t);
}
