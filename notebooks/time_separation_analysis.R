atac_data <- read.csv("~/PycharmProjects/domain_translation/data/a549_dex/processed/atac_data.csv", row.names=1)
atac_labels <- as.factor(atac_data$label)
atac_data <- atac_data[,1:NCOL(atac_data)-1]

rna_data <- read.csv("~/PycharmProjects/domain_translation/data/a549_dex/processed/rna_data.csv", row.names=1)
rna_labels <- as.factor(rna_data$label)
rna_data <- rna_data[,1:NCOL(rna_data)-1]

library(randomForest)
set.seed(1234)
test_idc <- sample(1:NROW(rna_data), size = round(0.2 * NROW(rna_data)), replace=F)
train_idc <- seq(1,NROW(rna_data))[-test_idc]
train_idc <- sample(train_idc, length(train_idc), replace = F)

atac_rf <- randomForest(x=atac_data[train_idc, ], y=atac_labels[train_idc], xtest=atac_data[test_idc, ], ytest=atac_labels[test_idc], importance=T, ntree=1000)
atac_rf$test$confusion


rna_rf <- randomForest(x=rna_data[train_idc, ], y=rna_labels[train_idc], xtest=rna_data[test_idc, ], ytest=rna_labels[test_idc], importance=T, ntree=1000)
rna_rf$test$confusion


# RF with PCA
pca_atac_data <- prcomp(atac_data, center=T, scale=T, rank=50)
pca_rna_data <- prcomp(rna_data, center=T, scale=T, rank=50)

atac_pca_rf <- randomForest(x=pca_atac_data$x[train_idc, ], y=atac_labels[train_idc], xtest=pca_atac_data$x[test_idc, ], ytest=atac_labels[test_idc], importance=T, ntree=1000)
rna_pca_rf <- randomForest(x=pca_rna_data$x[train_idc, ], y=rna_labels[train_idc], xtest=pca_rna_data$x[test_idc, ], ytest=rna_labels[test_idc], importance=T, ntree=1000)

atac_pca_rf$test$confusion
rna_pca_rf$test$confusion
