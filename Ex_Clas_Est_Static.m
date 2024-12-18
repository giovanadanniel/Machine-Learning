%% Database

load("DataSet.mat");
Targets = DataSet(:,end);
Inputs = DataSet(:,1:end-1);

% Visualização rápida dos dados
disp('Inputs:');
disp(Inputs(1:5, :)); % Mostrar as primeiras 5 linhas de inputs
disp('Targets:');
disp(Targets(1:5)); % Mostrar as primeiras 5 linhas de targets

%% Contagem de instâncias e características

Instances = size(Inputs, 1);
Features = size(Inputs, 2);

fprintf('Number of Instances: %d\n', Instances);
fprintf('Number of Features: %d\n', Features);

%% Calcular o balanceamento das classes

classLabels = unique(Targets);
classCounts = histc(Targets, classLabels);

for i = 1:length(classLabels)
    fprintf('Class %d: %d instances\n', classLabels(i), classCounts(i));
end

%% Verificar dados faltantes (NaN)

numMissing = sum(isnan(Inputs));

disp('Número de valores faltantes (NaN) por característica:');
for i = 1:Features
    fprintf('Feature %d: %d valores faltantes\n', i, numMissing(i));
end

rowsWithNaN = any(isnan(Inputs), 2);
numRowsWithNaN = sum(rowsWithNaN);

fprintf('Número de filas com valores NaN: %d\n', numRowsWithNaN);

%% Calcular matriz de correlação de Spearman

correlationMatrix = corr(Inputs, 'Type', 'Spearman');

disp('Spearman Correlation Matrix:');
disp(correlationMatrix);

figure;
heatmap(correlationMatrix, 'Colormap', jet, 'ColorbarVisible', 'on');
title('Spearman Correlation Matrix');
xlabel('Features');
ylabel('Features');

%% Criar boxplots por classe para cada característica

numFeatures = size(Inputs, 2);

figure;
for i = 1:numFeatures
    subplot(2, 5, i);
    boxplot(Inputs(:, i), Targets, 'Labels', {'Classe 1', 'Classe 2'});
    title(['Feature Boxplot', num2str(i)]);
    xlabel('Class');
    ylabel(['Feature ', num2str(i)]);
end
sgtitle('Comparison Feature x Class');

%% Calcular valores máximos, mínimos, médios e desvio padrão para cada característica

maxValues = max(Inputs);
minValues = min(Inputs);
meanValues = mean(Inputs);
stdValues = std(Inputs);

disp('Valores máximos, mínimos, médios e desvio padrão para cada característica:');
for i = 1:numFeatures
    fprintf('Feature %d: Máximo = %.4f, Mínimo = %.4f, Média = %.4f, Desvio padrão = %.4f\n', ...
        i, maxValues(i), minValues(i), meanValues(i), stdValues(i));
end

%% Criar histogramas para cada característica

figure;
for i = 1:numFeatures
    subplot(2, 5, i);
    histogram(Inputs(:, i));
    title(['Histograma de Feature ', num2str(i)]);
    xlabel(['Feature ', num2str(i)]);
    ylabel('Frequência');
end
sgtitle('Distribuição de cada característica');

%% Identificação de outliers utilizando o método de Tukey (1.5 IQR)

outliers = zeros(size(Inputs));
for i = 1:numFeatures
    Q1 = quantile(Inputs(:, i), 0.25);
    Q3 = quantile(Inputs(:, i), 0.75);
    IQR = Q3 - Q1;
    outlierIndices = (Inputs(:, i) < Q1 - 1.5 * IQR) | (Inputs(:, i) > Q3 + 1.5 * IQR);
    outliers(:, i) = outlierIndices;
end

disp('Número de outliers por característica:');
for i = 1:numFeatures
    fprintf('Feature %d: %d outliers\n', i, sum(outliers(:, i)));
end

%% Dividir os dados em treinamento (70%) e teste (30%) utilizando hold-out simples

cv = cvpartition(Targets, 'HoldOut', 0.3);

Inp_Train = Inputs(training(cv), :);
Tar_Train = Targets(training(cv));
Inp_Test = Inputs(test(cv), :);
Tar_Test = Targets(test(cv));

trainClassLabels = unique(Tar_Train);
trainClassCounts = histc(Tar_Train, trainClassLabels);

disp('Balanceamento de classes no conjunto de treinamento:');
for i = 1:length(trainClassLabels)
    fprintf('Classe %d: %d instâncias\n', trainClassLabels(i), trainClassCounts(i));
end

testClassLabels = unique(Tar_Test);
testClassCounts = histc(Tar_Test, testClassLabels);

disp('Balanceamento de classes no conjunto de teste:');
for i = 1:length(testClassLabels)
    fprintf('Classe %d: %d instâncias\n', testClassLabels(i), testClassCounts(i));
end

%% Configurar k-fold cross-validation com 5 pliegues

k = 5;
cv = cvpartition(Tar_Train, 'KFold', k);

models = {'kNN', 'Naive Bayes', 'SVM'};

accuracyArray = zeros(k, length(models));
recallArray = zeros(k, length(models));
precisionArray = zeros(k, length(models));
f1Array = zeros(k, length(models));
misclassificationArray = zeros(k, length(models));

for modelIdx = 1:length(models)
    for i = 1:k
        trainIdx = training(cv, i);
        testIdx = test(cv, i);
        Inp_Train_k = Inp_Train(trainIdx, :);
        Tar_Train_k = Tar_Train(trainIdx);
        Inp_Test_k = Inp_Train(testIdx, :);
        Tar_Test_k = Tar_Train(testIdx);

        switch models{modelIdx}
            case 'kNN'
                model = fitcknn(Inp_Train_k, Tar_Train_k);
            case 'Naive Bayes'
                model = fitcnb(Inp_Train_k, Tar_Train_k);
            case 'SVM'
                model = fitcsvm(Inp_Train_k, Tar_Train_k);
        end

        predictions = predict(model, Inp_Test_k);

        confMatrix = confusionmat(Tar_Test_k, predictions);
        TP = confMatrix(1, 1);
        FP = confMatrix(1, 2);
        FN = confMatrix(2, 1);
        TN = confMatrix(2, 2);

        accuracyArray(i, modelIdx) = (TP + TN) / (TP + TN + FP + FN);
        recallArray(i, modelIdx) = TP / (TP + FN);  % Para classe positiva
        precisionArray(i, modelIdx) = TP / (TP + FP); % Para classe positiva
        f1Array(i, modelIdx) = 2 * (precisionArray(i, modelIdx) * recallArray(i, modelIdx)) / ...
                                (precisionArray(i, modelIdx) + recallArray(i, modelIdx));
        misclassificationArray(i, modelIdx) = (FP + FN) / (TP + TN + FP + FN);
    end
end

for modelIdx = 1:length(models)
    fprintf('\nResultados de K-Fold Cross-Validation para o modelo %s (k = %d):\n', models{modelIdx}, k);
    fprintf('Accuracy: Mean = %.4f, Std = %.4f\n', mean(accuracyArray(:, modelIdx)), std(accuracyArray(:, modelIdx)));
    fprintf('Misclassification: Mean = %.4f, Std = %.4f\n', mean(misclassificationArray(:, modelIdx)), std(misclassificationArray(:, modelIdx)));
    fprintf('Recall: Mean = %.4f, Std = %.4f\n', mean(recallArray(:, modelIdx)), std(recallArray(:, modelIdx)));
    fprintf('Precision: Mean = %.4f, Std = %.4f\n', mean(precisionArray(:, modelIdx)), std(precisionArray(:, modelIdx)));
    fprintf('F1 Score: Mean = %.4f, Std = %.4f\n', mean(f1Array(:, modelIdx)), std(f1Array(:, modelIdx)));
end

%% Matriz de confusão e PLOT para cada modelo

figure;
for modelIdx = 1:length(models)
    switch models{modelIdx}
        case 'kNN'
            model = fitcknn(Inp_Train, Tar_Train);
        case 'Naive Bayes'
            model = fitcnb(Inp_Train, Tar_Train);
        case 'SVM'
            model = fitcsvm(Inp_Train, Tar_Train);
    end

    predictions = predict(model, Inp_Test);

    confMatrix = confusionmat(Tar_Test, predictions);
    
    fprintf('\nMatriz de Confusão para o modelo %s:\n', models{modelIdx});
    disp(confMatrix);

    subplot(1, 3, modelIdx);
    confusionchart(Tar_Test, predictions);
    title(sprintf('Matriz de Confusão (%s)', models{modelIdx}));

    if numFeatures >= 2
        figure;
        gscatter(Inp_Test(:,1), Inp_Test(:,2), predictions, 'rgb', 'osd');
        hold on;
        gscatter(Inp_Test(:,1), Inp_Test(:,2), Tar_Test, 'rgb', '+*x');
        title(sprintf('PLOT de %s', models{modelIdx}));
        legend('Predição Classe 1', 'Predição Classe 2', 'Predição Classe 3', ...
               'Verdadeiro Classe 1', 'Verdadeiro Classe 2', 'Verdadeiro Classe 3');
        xlabel('Feature 1');
        ylabel('Feature 2');
        hold off;
    end
end
