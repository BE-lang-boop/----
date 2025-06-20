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

// 蛮力法
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

// 动态规划法
void dp(int n, Item* items, int capacity, double* max_value, int* selected, int* can_record) {
    // 将价值转换为整数（乘以100）
    int* values_int = (int*)malloc(n * sizeof(int));
    if (values_int == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        values_int[i] = (int)(items[i].value * 100 + 0.5);
    }

    // 小规模且容量适中时使用二维数组记录方案
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

        // 动态规划填表
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

        // 回溯记录方案
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

        // 释放内存
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

// 比较函数
int compare_items(const void* a, const void* b) {
    Item* itemA = (Item*)a;
    Item* itemB = (Item*)b;
    double ratio_diff = itemB->ratio - itemA->ratio;
    if (ratio_diff > 0) return 1;
    if (ratio_diff < 0) return -1;
    return 0;
}

// 贪心法
void greedy(int n, Item* items, int capacity, double* max_value, int* selected) {
    *max_value = 0.0;
    int cur_weight = 0;

    // 复制物品数组以保留原始顺序
    Item* sorted_items = (Item*)malloc(n * sizeof(Item));
    if (sorted_items == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }
    memcpy(sorted_items, items, n * sizeof(Item));

    // 按价值/重量比降序排序
    qsort(sorted_items, n, sizeof(Item), compare_items);

    memset(selected, 0, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        if (cur_weight + sorted_items[i].weight <= capacity) {
            cur_weight += sorted_items[i].weight;
            *max_value += sorted_items[i].value;

            // 在原始物品数组中找到对应的物品并标记
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

// 回溯法
void backtracking(int n, Item* items, int capacity, double* max_value, int* selected, int* solution, int depth, int cur_weight, double cur_value) {
    if (depth == n) {
        if (cur_value > *max_value) {
            *max_value = cur_value;
            memcpy(solution, selected, n * sizeof(int));
        }
        return;
    }

    // 剪枝
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

    // 不选当前物品
    selected[depth] = 0;
    backtracking(n, items, capacity, max_value, selected, solution, depth + 1, cur_weight, cur_value);

    // 选当前物品
    if (cur_weight + items[depth].weight <= capacity) {
        selected[depth] = 1;
        backtracking(n, items, capacity, max_value, selected, solution, depth + 1,
            cur_weight + items[depth].weight, cur_value + items[depth].value);
        selected[depth] = 0;
    }
}

// 回溯法包装函数
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

// 生成随机物品
void generate_items(int n, Item* items) {
    for (int i = 0; i < n; i++) {
        items[i].id = i;
        items[i].weight = rand() % 100 + 1;  // 随机重量
        items[i].value = (rand() % 9001) / 100.0 + 100;  // 随机价值
        items[i].ratio = items[i].value / items[i].weight;
    }
}

// 打印选择的物品信息
void print_selected_items(int n, Item* items, int* selected, double max_value, const char* algorithm_name, int capacity) {
    int total_weight = 0;
    double total_value = 0.0;
    int count = 0;

    printf("\n%s 选择的物品 (背包容量: %d):\n", algorithm_name, capacity);
    printf("%-10s %-10s %-15s\n", "物品编号", "重量", "价值");
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
    printf("总件数: %d\n", count);
    printf("总重量: %d\n", total_weight);
    printf("总价值: %.2f\n", total_value);
    printf("计算的最大价值: %.2f\n", max_value);
    printf("验证: %s\n", fabs(total_value - max_value) < 0.01 ? "通过" : "失败");
    printf("\n");
}

// 保存物品信息到CSV文件
void save_item_info(int n, Item* items, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    fprintf(file, "物品编号,物品重量,物品价值\n");
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d,%d,%.2f\n",
            items[i].id + 1,
            items[i].weight,
            items[i].value);
    }

    fclose(file);
    printf("物品信息已保存到: %s\n", filename);
}

// 计时函数
double get_time_ms() {
    LARGE_INTEGER frequency, time;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart * 1000.0 / frequency.QuadPart;
}

// 测试小规模示例
void test_small_example() {
    printf("\n=============================================\n");
    printf("测试小规模示例 (5个物品, 背包容量10)\n");
    printf("=============================================\n");

    int n = 5;
    int capacity = 10;

    // 固定示例数据
    Item items[5] = {
        {0, 2, 6.00, 6.00 / 2},
        {1, 2, 3.00, 3.00 / 2},
        {2, 6, 5.00, 5.00 / 6},
        {3, 5, 4.00, 4.00 / 5},
        {4, 4, 6.00, 6.00 / 4}
    };

    // 蛮力法
    double max_value_bf = 0.0;
    int* selected_bf = (int*)calloc(n, sizeof(int));
    brute_force(n, items, capacity, &max_value_bf, selected_bf);
    printf("蛮力法: 最大价值 = %.2f\n", max_value_bf);
    print_selected_items(n, items, selected_bf, max_value_bf, "蛮力法", capacity);
    free(selected_bf);

    // 动态规划法
    double max_value_dp = 0.0;
    int* selected_dp = (int*)calloc(n, sizeof(int));
    int can_record = 0;
    dp(n, items, capacity, &max_value_dp, selected_dp, &can_record);
    printf("动态规划法: 最大价值 = %.2f\n", max_value_dp);
    print_selected_items(n, items, selected_dp, max_value_dp, "动态规划法", capacity);
    free(selected_dp);

    // 贪心法
    double max_value_gr = 0.0;
    int* selected_gr = (int*)calloc(n, sizeof(int));
    greedy(n, items, capacity, &max_value_gr, selected_gr);
    printf("贪心法: 最大价值 = %.2f\n", max_value_gr);
    print_selected_items(n, items, selected_gr, max_value_gr, "贪心法", capacity);
    free(selected_gr);

    // 回溯法
    double max_value_bt = 0.0;
    int* selected_bt = (int*)calloc(n, sizeof(int));
    backtracking_wrapper(n, items, capacity, &max_value_bt, selected_bt);
    printf("回溯法: 最大价值 = %.2f\n", max_value_bt);
    print_selected_items(n, items, selected_bt, max_value_bt, "回溯法", capacity);
    free(selected_bt);
}

int main() {
    srand((unsigned int)time(NULL));  // 初始化随机种子

    // 测试小规模示例
    test_small_example();

    long n_list[] = { 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 40000, 80000, 160000, 320000 };
    int capacity_list[] = { 10000, 100000, 1000000 };
    int num_n = sizeof(n_list) / sizeof(n_list[0]);
    int num_capacities = sizeof(capacity_list) / sizeof(capacity_list[0]);

    // 创建结果CSV文件
    FILE* output = fopen("results.csv", "w");
    if (output == NULL) {
        perror("Error opening file");
        return 1;
    }
    fprintf(output, "算法,物品数量,背包容量,最大价值,执行时间(ms)\n");

    // 创建物品信息文件
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

            // 生成物品数据
            double gen_start = get_time_ms();
            generate_items(n, items);
            double gen_end = get_time_ms();

            // 保存1000个物品的信息
            if (n == 1000 && !items_1000_generated) {
                char filename[50];
                sprintf(filename, "items_1000_cap_%d.csv", capacity);
                save_item_info(n, items, filename);
                items_1000_generated = 1;
            }

            printf("\n=============================================\n");
            printf("物品数量: %ld, 背包容量: %d\n", n, capacity);
            printf("数据生成时间: %.2f ms\n", gen_end - gen_start);
            printf("=============================================\n");

            // 对于小规模数据，运行所有算法
            if (n <= 20) {
                // 蛮力法
                double start_bf = get_time_ms();
                double max_value_bf = 0.0;
                int* selected_bf = (int*)calloc(n, sizeof(int));
                if (selected_bf) {
                    brute_force(n, items, capacity, &max_value_bf, selected_bf);
                    double end_bf = get_time_ms();
                    double time_bf = end_bf - start_bf;

                    fprintf(output, "蛮力法,%ld,%d,%.2f,%.2f\n", n, capacity, max_value_bf, time_bf);
                    printf("蛮力法: 最大价值 = %.2f, 执行时间 = %.2f ms\n", max_value_bf, time_bf);

                    // 打印选择的物品信息
                    print_selected_items(n, items, selected_bf, max_value_bf, "蛮力法", capacity);
                    free(selected_bf);
                }

                // 回溯法
                double start_bt = get_time_ms();
                double max_value_bt = 0.0;
                int* selected_bt = (int*)calloc(n, sizeof(int));
                if (selected_bt) {
                    backtracking_wrapper(n, items, capacity, &max_value_bt, selected_bt);
                    double end_bt = get_time_ms();
                    double time_bt = end_bt - start_bt;

                    fprintf(output, "回溯法,%ld,%d,%.2f,%.2f\n", n, capacity, max_value_bt, time_bt);
                    printf("回溯法: 最大价值 = %.2f, 执行时间 = %.2f ms\n", max_value_bt, time_bt);

                    // 打印选择的物品信息
                    print_selected_items(n, items, selected_bt, max_value_bt, "回溯法", capacity);
                    free(selected_bt);
                }
            }

            // 动态规划法
            double start_dp = get_time_ms();
            double max_value_dp = 0.0;
            int* selected_dp = (int*)calloc(n, sizeof(int));
            int can_record = 0;
            if (selected_dp) {
                dp(n, items, capacity, &max_value_dp, selected_dp, &can_record);
                double end_dp = get_time_ms();
                double time_dp = end_dp - start_dp;

                fprintf(output, "动态规划法,%ld,%d,%.2f,%.2f\n", n, capacity, max_value_dp, time_dp);
                printf("动态规划法: 最大价值 = %.2f, 执行时间 = %.2f ms\n", max_value_dp, time_dp);

                // 当n≤20时打印选择的物品信息
                if (n <= 20) {
                    print_selected_items(n, items, selected_dp, max_value_dp, "动态规划法", capacity);
                }
                free(selected_dp);
            }

            // 贪心法
            double start_gr = get_time_ms();
            double max_value_gr = 0.0;
            int* selected_gr = (int*)calloc(n, sizeof(int));
            if (selected_gr) {
                greedy(n, items, capacity, &max_value_gr, selected_gr);
                double end_gr = get_time_ms();
                double time_gr = end_gr - start_gr;

                fprintf(output, "贪心法,%ld,%d,%.2f,%.2f\n", n, capacity, max_value_gr, time_gr);
                printf("贪心法: 最大价值 = %.2f, 执行时间 = %.2f ms\n", max_value_gr, time_gr);

                // 当n≤20时打印选择的物品信息
                if (n <= 20) {
                    print_selected_items(n, items, selected_gr, max_value_gr, "贪心法", capacity);
                }
                free(selected_gr);
            }

            free(items);
            printf("\n");
        }
    }

    fclose(output);
    printf("所有测试完成! 结果已保存到 results.csv\n");
    return 0;
}