from torch import nn
from layers.ReTATSF import TS_CoherAnalysis, ContentSynthesis, TextCrossAttention, CrossandOutput

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.TS_CoherAnalysis = TS_CoherAnalysis(configs)
        self.ContentSynthesis = ContentSynthesis(configs)

        self.TextCrossAttention = TextCrossAttention(configs)
        self.CrossandOutput = CrossandOutput(configs)

        self.configs = configs

    def forward(self, target_series, TS_database, qt, des, newsdatabase):
        #Time Series
        ref_TS = self.TS_CoherAnalysis(target_series, TS_database)#[B, C_T*K_T, L]
        TS_Synthesis = self.ContentSynthesis(target_series, ref_TS) #[B, C_T*(K_T+1), L, D]

        #Text
        Text_Synthesis = self.TextCrossAttention(qt, des, newsdatabase)#[B, C_T*K_n, H, D]

        #Cross and Output
        prediction = self.CrossandOutput(Text_Synthesis, TS_Synthesis) #[B, C_T, H]

        return prediction #[B, 1, L]