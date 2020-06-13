library(ggplot2)
library(dplyr)
library(tidyverse)
methods_df = data.frame(method=c("BaseGenCF", "AEGenCF", "ModelApproxGenCF", "OracleGenCF", "SCMGenCF", "CEM"), 
                        method2=c("Variational", "AEGenCF", "ModelApprox", "ExampleBased", "ModelBased", "Baseline (CEM)"))
df = read.csv("plotdf.csv")
df = filter(df, !(method %in% c("AEGenCF", "BaseGenCF")))
df = inner_join(df, methods_df, by="method")
df$method=df$method2
df$dataset = factor(df$dataset, levels=c("bn1", "sangiovese", "adult-age", "adult-age-ed"))
df$method = factor(df$method, levels=c("Variational", "AEGenCF", "ModelBased", "ModelApprox","ExampleBased",  "Baseline (CEM)"))
df$y = ifelse(df$y==0, NA, df$y)

# plot for constraint score
constraint_df = filter(df, metric=="const-score")
constraint_df = mutate(constraint_df, linetype=ifelse(method %in% c("Baseline (CEM)", "AEGenCF", "BaseGenCF"), "s", "d"))
constraint_df = mutate(constraint_df, linesize=ifelse(method %in% c("ExampleBased", "ModelApprox"), "main","other"))

ggplot(constraint_df, aes(x=dataset,y=y,group=method,color=method,shape=method)) +
  geom_point(size=3) + geom_line(aes(linetype=linetype,size=linesize))+
  geom_errorbar(aes(ymin=y-err, ymax=y+err, width=0.2)) +
  scale_size_manual(values=c(1.38,1))+
  scale_color_manual(values=c("#F8766D", "#7CAE00", "#00BFC4", "#708090"))+
  ylim(0,100)+
  guides(size=FALSE)+
  guides(linetype=FALSE)+
  guides(color=guide_legend(nrow=2))+
  theme_bw() +
  theme(axis.text.x=element_text(angle=325, size=12),
        axis.text.y=element_text(size=16),
        axis.title.y=element_text(size=14),
        legend.title=element_blank(),
        legend.position="bottom",
        legend.text=element_text(size=10))+
  ylab("Constraint Score") +
  xlab("")
ggsave("results/constraint-score.pdf", width=3, height=4)

library(scales)
hue_pal()(4)

# causal-edge-score
causal_df = filter(df, metric=="dist-score")
causal_df = mutate(causal_df, 
                   linetype=ifelse(method %in% c("Baseline (CEM)", "AEGenCF", "BaseGenCF"), "s", "d"),
                   linesize=ifelse(method %in% c("ExampleBased", "ModelApprox"), "main","other"))
causal_df$y = ifelse(causal_df$dataset=="bn1", causal_df$y/20, causal_df$y)
ggplot(causal_df, aes(x=dataset,y=y,group=method,color=method,shape=method)) +
  geom_point(size=3) + geom_line(aes(linetype=linetype,size=linesize))+
  geom_errorbar(aes(ymin=y-err, ymax=y+err, width=0.1)) +
  scale_size_manual(values=c(1.38,1))+
  scale_color_manual(values=c("#F8766D", "#7CAE00", "#00BFC4", "#708090"))+
  ylim(-4,4)+
  guides(size=FALSE)+
  guides(linetype=FALSE)+
  guides(color=guide_legend(nrow=2))+
  theme_bw() +
  theme(axis.text.x=element_text(angle=325, size=12),
        axis.text.y=element_text(size=16),
        axis.title.y=element_text(size=14),
        legend.title=element_blank(),
        legend.position="bottom",
        legend.text=element_text(size=10))+
  ylab("Causal-Edge Score") +
  xlab("")
ggsave("results/causaledge-score.pdf", width=3, height=4)

