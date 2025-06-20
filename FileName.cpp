#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>  

typedef struct {
    int id;
    int weight;
    double value;
    double ratio; 
} Item;

void brute_force(int n, Item* items, int capacity, double* max_value, int* solution);
void dp(int n, Item* items, int capacity, double* max_value, int* selected, int* can_record);
void greedy(int n, Item* items, int capacity, double* max_value, int* selected);
void backtracking(int n, Item* items, int capacity, double* max_value, int* selected, int* solution, int depth, int cur_weight, double cur_value);
void backtracking_wrapper(int n, Item* items, int capacity, double* max_value, int* solution);
void generate_items(int n, Item* items);
void print_selected_items(int n, Item* items, int* selected, double max_value, const char* algorithm_name, int capacity);
void save_item_info(int n, Item* items, const char* filename);
void test_small_example();

// ������
void brute_force(int n, Item* items, int capacity, double* max_value, int* solution) {
    *max_value = 0;
    uint64_t total = 1ULL << n;  
    uint64_t best_mask = 0;

    for (uint64_t mask = 0; mask < total; mask++) {
        int cur_weight = 0;
        double cur_value = 0.0;
        for (int i = 0; i < n; i++) {
            if (mask & (1ULL << i)) {
                cur_weight += items[i].weight;
                cur_value += items[i].value;
            }
        }
        if (cur_weight <= capacity && cur_value > *max_value) {
            *max_value = cur_value;
            best_mask = mask;
        }
    }

    for (int i = 0; i < n; i++) {
        solution[i] = (best_mask & (1ULL << i)) ? 1 : 0;
    }
}

// ��̬�滮��
void dp(int n, Item* items, int capacity, double* max_value, int* selected, int* can_record) {
    // ����ֵת��Ϊ����������100��
    int* values_int = (int*)malloc(n * sizeof(int));
    if (values_int == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        values_int[i] = (int)(items[i].value * 100 + 0.5);
    }

    // С��ģ����������ʱʹ�ö�ά�����¼����
    if (n <= 1000 && capacity <= 10000) {
        *can_record = 1;
        long long** dp = (long long**)malloc((n + 1) * sizeof(long long*));
        if (dp == NULL) {
            perror("Memory allocation failed");
            exit(1);
        }

        for (int i = 0; i <= n; i++) {
            dp[i] = (long long*)malloc((capacity + 1) * sizeof(long long));
            if (dp[i] == NULL) {
                perror("Memory allocation failed");
                exit(1);
            }
            memset(dp[i], 0, (capacity + 1) * sizeof(long long));
        }

        // ��̬�滮���
        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                dp[i][w] = dp[i - 1][w];
                if (w >= items[i - 1].weight) {
                    long long new_val = dp[i - 1][w - items[i - 1].weight] + values_int[i - 1];
                    if (new_val > dp[i][w]) {
                        dp[i][w] = new_val;
                    }
                }
            }
        }

        *max_value = dp[n][capacity] / 100.0;

        // ���ݼ�¼����
        int w = capacity;
        for (int i = n; i > 0; i--) {
            if (dp[i][w] != dp[i - 1][w] && w >= items[i - 1].weight) {
                selected[i - 1] = 1;
                w -= items[i - 1].weight;
            }
            else {
                selected[i - 1] = 0;
            }
        }

        // �ͷ��ڴ�
        for (int i = 0; i <= n; i++) free(dp[i]);
        free(dp);
    }
    else {
        *can_record = 0;
        long long* dp = (long long*)calloc(capacity + 1, sizeof(long long));
        if (dp == NULL) {
            perror("Memory allocation failed");
            exit(1);
        }
        for (int i = 0; i < n; i++) {
            for (int w = capacity; w >= items[i].weight; w--) {
                long long new_val = dp[w - items[i].weight] + values_int[i];
                if (new_val > dp[w]) {
                    dp[w] = new_val;
                }
            }
        }
        *max_value = dp[capacity] / 100.0;
        free(dp);
    }
    free(values_int);
}

// �ȽϺ���
int compare_items(const void* a, const void* b) {
    Item* itemA = (Item*)a;
    Item* itemB = (Item*)b;
    double ratio_diff = itemB->ratio - itemA->ratio;
    if (ratio_diff > 0) return 1;
    if (ratio_diff < 0) return -1;
    return 0;
}

// ̰�ķ�
void greedy(int n, Item* items, int capacity, double* max_value, int* selected) {
    *max_value = 0.0;
    int cur_weight = 0;

    // ������Ʒ�����Ա���ԭʼ˳��
    Item* sorted_items = (Item*)malloc(n * sizeof(Item));
    if (sorted_items == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }
    memcpy(sorted_items, items, n * sizeof(Item));

    // ����ֵ/�����Ƚ�������
    qsort(sorted_items, n, sizeof(Item), compare_items);

    memset(selected, 0, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        if (cur_weight + sorted_items[i].weight <= capacity) {
            cur_weight += sorted_items[i].weight;
            *max_value += sorted_items[i].value;

            // ��ԭʼ��Ʒ�������ҵ���Ӧ����Ʒ�����
            for (int j = 0; j < n; j++) {
                if (items[j].id == sorted_items[i].id) {
                    selected[j] = 1;
                    break;
                }
            }
        }
    }
    free(sorted_items);
}

// ���ݷ�
void backtracking(int n, Item* items, int capacity, double* max_value, int* selected, int* solution, int depth, int cur_weight, double cur_value) {
    if (depth == n) {
        if (cur_value > *max_value) {
            *max_value = cur_value;
            memcpy(solution, selected, n * sizeof(int));
        }
        return;
    }

    // ��֦
    double bound = cur_value;
    int bound_weight = cur_weight;
    for (int i = depth; i < n && bound_weight <= capacity; i++) {
        if (bound_weight + items[i].weight <= capacity) {
            bound += items[i].value;
            bound_weight += items[i].weight;
        }
        else {
            double fraction = (capacity - bound_weight) / (double)items[i].weight;
            bound += items[i].value * fraction;
            break;
        }
    }

    if (bound <= *max_value) return;

    // ��ѡ��ǰ��Ʒ
    selected[depth] = 0;
    backtracking(n, items, capacity, max_value, selected, solution, depth + 1, cur_weight, cur_value);

    // ѡ��ǰ��Ʒ
    if (cur_weight + items[depth].weight <= capacity) {
        selected[depth] = 1;
        backtracking(n, items, capacity, max_value, selected, solution, depth + 1,
            cur_weight + items[depth].weight, cur_value + items[depth].value);
        selected[depth] = 0;
    }
}

// ���ݷ���װ����
void backtracking_wrapper(int n, Item* items, int capacity, double* max_value, int* solution) {
    *max_value = 0.0;
    int* selected = (int*)calloc(n, sizeof(int));
    if (selected == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }
    backtracking(n, items, capacity, max_value, selected, solution, 0, 0, 0.0);
    free(selected);
}

// ���������Ʒ
void generate_items(int n, Item* items) {
    for (int i = 0; i < n; i++) {
        items[i].id = i;
        items[i].weight = rand() % 100 + 1;  // �������
        items[i].value = (rand() % 9001) / 100.0 + 100;  // �����ֵ
        items[i].ratio = items[i].value / items[i].weight;
    }
}

// ��ӡѡ�����Ʒ��Ϣ
void print_selected_items(int n, Item* items, int* selected, double max_value, const char* algorithm_name, int capacity) {
    int total_weight = 0;
    double total_value = 0.0;
    int count = 0;

    printf("\n%s ѡ�����Ʒ (��������: %d):\n", algorithm_name, capacity);
    printf("%-10s %-10s %-15s\n", "��Ʒ���", "����", "��ֵ");
    printf("---------------------------------\n");

    for (int i = 0; i < n; i++) {
        if (selected[i]) {
            printf("%-10d %-10d %-15.2f\n",
                items[i].id + 1,
                items[i].weight,
                items[i].value);
            total_weight += items[i].weight;
            total_value += items[i].value;
            count++;
        }
    }

    printf("---------------------------------\n");
    printf("�ܼ���: %d\n", count);
    printf("������: %d\n", total_weight);
    printf("�ܼ�ֵ: %.2f\n", total_value);
    printf("���������ֵ: %.2f\n", max_value);
    printf("��֤: %s\n", fabs(total_value - max_value) < 0.01 ? "ͨ��" : "ʧ��");
    printf("\n");
}

// ������Ʒ��Ϣ��CSV�ļ�
void save_item_info(int n, Item* items, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    fprintf(file, "��Ʒ���,��Ʒ����,��Ʒ��ֵ\n");
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d,%d,%.2f\n",
            items[i].id + 1,
            items[i].weight,
            items[i].value);
    }

    fclose(file);
    printf("��Ʒ��Ϣ�ѱ��浽: %s\n", filename);
}

// ��ʱ����
double get_time_ms() {
    LARGE_INTEGER frequency, time;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart * 1000.0 / frequency.QuadPart;
}

// ����С��ģʾ��
void test_small_example() {
    printf("\n=============================================\n");
    printf("����С��ģʾ�� (5����Ʒ, ��������10)\n");
    printf("=============================================\n");

    int n = 5;
    int capacity = 10;

    // �̶�ʾ������
    Item items[5] = {
        {0, 2, 6.00, 6.00 / 2},
        {1, 2, 3.00, 3.00 / 2},
        {2, 6, 5.00, 5.00 / 6},
        {3, 5, 4.00, 4.00 / 5},
        {4, 4, 6.00, 6.00 / 4}
    };

    // ������
    double max_value_bf = 0.0;
    int* selected_bf = (int*)calloc(n, sizeof(int));
    brute_force(n, items, capacity, &max_value_bf, selected_bf);
    printf("������: ����ֵ = %.2f\n", max_value_bf);
    print_selected_items(n, items, selected_bf, max_value_bf, "������", capacity);
    free(selected_bf);

    // ��̬�滮��
    double max_value_dp = 0.0;
    int* selected_dp = (int*)calloc(n, sizeof(int));
    int can_record = 0;
    dp(n, items, capacity, &max_value_dp, selected_dp, &can_record);
    printf("��̬�滮��: ����ֵ = %.2f\n", max_value_dp);
    print_selected_items(n, items, selected_dp, max_value_dp, "��̬�滮��", capacity);
    free(selected_dp);

    // ̰�ķ�
    double max_value_gr = 0.0;
    int* selected_gr = (int*)calloc(n, sizeof(int));
    greedy(n, items, capacity, &max_value_gr, selected_gr);
    printf("̰�ķ�: ����ֵ = %.2f\n", max_value_gr);
    print_selected_items(n, items, selected_gr, max_value_gr, "̰�ķ�", capacity);
    free(selected_gr);

    // ���ݷ�
    double max_value_bt = 0.0;
    int* selected_bt = (int*)calloc(n, sizeof(int));
    backtracking_wrapper(n, items, capacity, &max_value_bt, selected_bt);
    printf("���ݷ�: ����ֵ = %.2f\n", max_value_bt);
    print_selected_items(n, items, selected_bt, max_value_bt, "���ݷ�", capacity);
    free(selected_bt);
}

int main() {
    srand((unsigned int)time(NULL));  // ��ʼ���������

    // ����С��ģʾ��
    test_small_example();

    long n_list[] = { 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 40000, 80000, 160000, 320000 };
    int capacity_list[] = { 10000, 100000, 1000000 };
    int num_n = sizeof(n_list) / sizeof(n_list[0]);
    int num_capacities = sizeof(capacity_list) / sizeof(capacity_list[0]);

    // �������CSV�ļ�
    FILE* output = fopen("results.csv", "w");
    if (output == NULL) {
        perror("Error opening file");
        return 1;
    }
    fprintf(output, "�㷨,��Ʒ����,��������,����ֵ,ִ��ʱ��(ms)\n");

    // ������Ʒ��Ϣ�ļ�
    int items_1000_generated = 0;

    for (int c_idx = 0; c_idx < num_capacities; c_idx++) {
        int capacity = capacity_list[c_idx];
        for (int n_idx = 0; n_idx < num_n; n_idx++) {
            long n = n_list[n_idx];
            Item* items = (Item*)malloc(n * sizeof(Item));
            if (items == NULL) {
                perror("Memory allocation failed");
                continue;
            }

            // ������Ʒ����
            double gen_start = get_time_ms();
            generate_items(n, items);
            double gen_end = get_time_ms();

            // ����1000����Ʒ����Ϣ
            if (n == 1000 && !items_1000_generated) {
                char filename[50];
                sprintf(filename, "items_1000_cap_%d.csv", capacity);
                save_item_info(n, items, filename);
                items_1000_generated = 1;
            }

            printf("\n=============================================\n");
            printf("��Ʒ����: %ld, ��������: %d\n", n, capacity);
            printf("��������ʱ��: %.2f ms\n", gen_end - gen_start);
            printf("=============================================\n");

            // ����С��ģ���ݣ����������㷨
            if (n <= 20) {
                // ������
                double start_bf = get_time_ms();
                double max_value_bf = 0.0;
                int* selected_bf = (int*)calloc(n, sizeof(int));
                if (selected_bf) {
                    brute_force(n, items, capacity, &max_value_bf, selected_bf);
                    double end_bf = get_time_ms();
                    double time_bf = end_bf - start_bf;

                    fprintf(output, "������,%ld,%d,%.2f,%.2f\n", n, capacity, max_value_bf, time_bf);
                    printf("������: ����ֵ = %.2f, ִ��ʱ�� = %.2f ms\n", max_value_bf, time_bf);

                    // ��ӡѡ�����Ʒ��Ϣ
                    print_selected_items(n, items, selected_bf, max_value_bf, "������", capacity);
                    free(selected_bf);
                }

                // ���ݷ�
                double start_bt = get_time_ms();
                double max_value_bt = 0.0;
                int* selected_bt = (int*)calloc(n, sizeof(int));
                if (selected_bt) {
                    backtracking_wrapper(n, items, capacity, &max_value_bt, selected_bt);
                    double end_bt = get_time_ms();
                    double time_bt = end_bt - start_bt;

                    fprintf(output, "���ݷ�,%ld,%d,%.2f,%.2f\n", n, capacity, max_value_bt, time_bt);
                    printf("���ݷ�: ����ֵ = %.2f, ִ��ʱ�� = %.2f ms\n", max_value_bt, time_bt);

                    // ��ӡѡ�����Ʒ��Ϣ
                    print_selected_items(n, items, selected_bt, max_value_bt, "���ݷ�", capacity);
                    free(selected_bt);
                }
            }

            // ��̬�滮��
            double start_dp = get_time_ms();
            double max_value_dp = 0.0;
            int* selected_dp = (int*)calloc(n, sizeof(int));
            int can_record = 0;
            if (selected_dp) {
                dp(n, items, capacity, &max_value_dp, selected_dp, &can_record);
                double end_dp = get_time_ms();
                double time_dp = end_dp - start_dp;

                fprintf(output, "��̬�滮��,%ld,%d,%.2f,%.2f\n", n, capacity, max_value_dp, time_dp);
                printf("��̬�滮��: ����ֵ = %.2f, ִ��ʱ�� = %.2f ms\n", max_value_dp, time_dp);

                // ��n��20ʱ��ӡѡ�����Ʒ��Ϣ
                if (n <= 20) {
                    print_selected_items(n, items, selected_dp, max_value_dp, "��̬�滮��", capacity);
                }
                free(selected_dp);
            }

            // ̰�ķ�
            double start_gr = get_time_ms();
            double max_value_gr = 0.0;
            int* selected_gr = (int*)calloc(n, sizeof(int));
            if (selected_gr) {
                greedy(n, items, capacity, &max_value_gr, selected_gr);
                double end_gr = get_time_ms();
                double time_gr = end_gr - start_gr;

                fprintf(output, "̰�ķ�,%ld,%d,%.2f,%.2f\n", n, capacity, max_value_gr, time_gr);
                printf("̰�ķ�: ����ֵ = %.2f, ִ��ʱ�� = %.2f ms\n", max_value_gr, time_gr);

                // ��n��20ʱ��ӡѡ�����Ʒ��Ϣ
                if (n <= 20) {
                    print_selected_items(n, items, selected_gr, max_value_gr, "̰�ķ�", capacity);
                }
                free(selected_gr);
            }

            free(items);
            printf("\n");
        }
    }

    fclose(output);
    printf("���в������! ����ѱ��浽 results.csv\n");
    return 0;
}