import pandas as pd
from typing import Union
from typing import Any
import numpy as np

watermelons = pd.read_csv('watermelon.csv')
watermelons.iloc[:, 0] = None

del watermelons['id']
del watermelons['density']
del watermelons['sugar']

# 朴素贝叶斯分类器
class NaiveBayesianClassifier:
    
    # 目标变量默认为最后 1 列
    # 第 1 列到倒数第 2 列是自变量
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    # target 为目标变量取值
    # 返回先验概率 Pr(target)
    def prior_prob(self, target: Union[str, int]) -> float:
        # 总变量个数
        n_var = self.data.shape[1]
        
        # 标签为 target 的记录的个数
        n_target = (self.data.iloc[:, n_var-1] == target).sum()
        
        # 样本量
        n_samples = self.data.shape[0]
        
        # 先验概率
        pr_target = n_target / n_samples
        
        return pr_target
    
    # target 为目标变量的取值
    # input_variables 为输入自变量向量
    # 输出为 Pr(target | input_variable) * Pr(input_variable)
    def probability(
        self,
        target: Any,
        input_variables: np.array
    ) -> float:
        
        # 先验概率
        pr_target = self.prior_prob(target)
        
        return pr_target
    
    # 根据条件独立性计算条件概率 Pr(input_variables | target)
    def cond_prob(
        self, 
        target: Any,
        input_variables: np.array
    ) -> float:
        col_idx = 0
        n_cols = self.data.shape[1]
        prob = 1
        while col_idx <= n_cols-2:
            prob = prob * self.single_cond_prob(
                col_idx, 
                input_variables[col_idx],
                target
            )
            col_idx = col_idx + 1
            
        return prob
    
    # 计算单个变量的条件概率 Pr(input_variable | target)
    def single_cond_prob(
        self, 
        col_idx: int, 
        input_variable: Any,
        target: Any
    ) -> float:
        
        n_cols = self.data.shape[1]
        last_col = self.data.iloc[:, n_cols-1]
        selector = last_col == target
        subset = self.data.loc[selector, :].iloc[:, col_idx]
        
        match_selector = subset == input_variable
        n_matches = match_selector.sum() or 1
        subset_size = subset.shape[0]
        
        return n_matches / subset_size
    
    # 根据输入预测所属类别
    def predict(self, input_variables: np.array) -> Any:
        
        n_cols = self.data.shape[1]
        unique_classes = self.data.iloc[:, n_cols-1].unique()
        odds = []
        for c in unique_classes:
            odds.append(
                self.cond_prob(c, input_variables)
            )
        odds_np = np.array(odds)
        max_idx = odds_np.argmax()
        return unique_classes[max_idx]

f = NaiveBayesianClassifier(watermelons)

row_idx = 0
n_cols = f.data.shape[1]
input_variables = f.data.iloc[row_idx, range(0, n_cols-1)]

print(f.predict(input_variables))

