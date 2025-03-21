import torch
from torch import nn
from layers.ReTATSF import TS_CoherAnalysis, ContentSynthesis, QueryTextencoder, TextCrossAttention, CrossandOutput

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.TS_CoherAnalysis = TS_CoherAnalysis(configs)
        self.ContentSynthesis = ContentSynthesis(configs)
        #self.QueryTextencoder = QueryTextencoder()
        self.TextCrossAttention = TextCrossAttention(configs)
        self.CrossandOutput = CrossandOutput(configs)

        self.configs = configs

    def forward(self, target_series, TS_database, qt, newsdatabase):
        #Time Series
        ref_TS = self.TS_CoherAnalysis(target_series, TS_database)#[B, nref, L]
        #print('ref_TS: ', ref_TS)
        TS_Synthesis = self.ContentSynthesis(target_series, ref_TS) #[B, K_temp+1, L, D_temp]

        #Text
        #qt_embedding = self.QueryTextencoder(qt).unsqueeze(1)#[B, H(1), D_text(384)]->[B,  K(1), H(1), D_text(384)]
        Text_Synthesis = self.TextCrossAttention(qt, newsdatabase)#[B, K_text, H, D_text]
        #print('Q: ', Text_Synthesis)

        #Cross and Output
        prediction = self.CrossandOutput(Text_Synthesis, TS_Synthesis) #[B, 1, L]

        return prediction #[B, 1, L]