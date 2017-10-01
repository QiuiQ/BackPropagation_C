// BackPropagation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "stdafx.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
using namespace std;

class BackPropagation                                                     //include OutputLayer
{
private:
	int layer, *synapse, *eachLayerNeuron, inputNeuron;//Each Layer Neuron
	float** value, ** weight, **bias, *inputLayerValue, *result, **err;//inputLayerValue = inputLayerValue

	void computErr()
	{
		for (int i = layer - 1; i >= 0; i--)
		{
			int sy = 0;
			if (i == layer - 1) for (int j = 0; j < eachLayerNeuron[i]; j++)
			{
				printf("ERR = 0.25 * (%f - %f)", value[i][j], value[i][j], result[0], value[i][j]);//result需要研究
				err[i][j] = value[i][j] * (1 - value[i][j]) * (result[0] - value[i][j]);//((result[j] - value[i][j]) * (result[j] - value[i][j]))/2;
				printf(" = %f\n", err[i][j]);
			}
			else
			for (int j = 0; j < eachLayerNeuron[i]; j++)
			{
				float temp = 0;
				for (int k = 0; k < eachLayerNeuron[i + 1]; k++) temp += err[i + 1][k] * weight[i + 1][sy++];
				err[i][j] = temp * value[i][j] * (1 - value[i][j]);
			}
		}
	}

public:
	BackPropagation(int l) : layer(l){}
	~BackPropagation()
	{
		delete eachLayerNeuron;
		delete inputLayerValue;
		delete synapse;
		delete value;
		delete weight;
		delete bias;
		delete err;
	}

	void initialNeuron(int inputN, int* n)
	{
		inputNeuron = inputN;
		eachLayerNeuron = n;
		inputLayerValue[inputNeuron];
		synapse = new int[layer];
		value = new float*[layer];
		weight = new float*[layer];
		bias = new float*[layer];
		err = new float*[layer];
		for (int i = 0; i < layer; i++)
		{
			value[i] = new float[eachLayerNeuron[i]];
			bias[i] = new float[eachLayerNeuron[i]];
			err[i] = new float[eachLayerNeuron[i]];
			if (i == 0)
			{
				weight[i] = new float[inputNeuron * eachLayerNeuron[i]];
				synapse[i] = inputNeuron * eachLayerNeuron[i];
			}
			else
			{
				synapse[i] = eachLayerNeuron[i - 1] * eachLayerNeuron[i];
				weight[i] = new float[eachLayerNeuron[i - 1] * eachLayerNeuron[i]];
			}
		}
		initialWeightBias();
	}

	void initialWeightBias()
	{
		srand(time(NULL));
		rand();
		for (int i = 0; i < layer; i++)
		{
			for (int j = 0; j < eachLayerNeuron[i]; j++) bias[i][j] = 0;// ((0.5 - ((float)rand() / RAND_MAX)));
			for (int j = 0; j < synapse[i]; j++) weight[i][j] = 0;// ((0.5 - ((float)rand() / RAND_MAX)));
		}
	}

	void inputValue(float* v) { inputLayerValue = v; }
	void inputTrueValue(int n, float* r)
	{
		result = new float[n];
		result = r;
		printf("\nTRUE VALUE = %f\n", result[0]);
	}

	void caculation()
	{
		for (int i = 0; i < layer; i++)
		{
			int sy = 0;//synapse
			if (i == 0)
			for (int j = 0; j < eachLayerNeuron[i]; j++)
			{
				float temp = 0;
				for (int k = 0; k < inputNeuron; k++)
				{
					//printf("%f*%f=%.9f\n", inputLayerValue[k], weight[i][sy], inputLayerValue[k] * weight[i][sy]);
					temp += inputLayerValue[k] * weight[i][sy++];
				}
				value[i][j] = 1.0 / (1.0 + exp(-1.0 * (temp + bias[i][j])));
				//printf("VALUE[%d][%d]=%.9f\n\n", i, j, value[i][j]);
			}
			else//目前用不到
			for (int j = 0; j < eachLayerNeuron[i]; j++)
			{
				float temp = 0;
				for (int k = 0; k < eachLayerNeuron[i - 1]; k++)
				{
					printf("%f*%f=%.9f\n", value[i - 1][k], weight[i][sy], value[i - 1][k] * weight[i][sy]);
					temp += value[i - 1][k] * weight[i][sy++];
				}
				value[i][j] = 1.0 / (1.0 + exp(-1.0 * (temp + bias[i][j])));
				printf("VALUE[%d][%d]=%.9f\n\n", i, j, value[i][j]);
			}
		}
		//cout << "預測結果 = " << value[layer - 1][0] << endl;
		printf("預測結果 = %.9f\n", value[layer - 1][0]);
		getchar();
		computErr();
	}


	void adjustWeightBias(float LEARNING_RATE)
	{
		for (int i = 0; i < layer; i++)
		{
			int sy = 0;//synapse
			if (i == 0)
			for (int j = 0; j < inputNeuron; j++)
			for (int k = 0; k < eachLayerNeuron[i]; k++)
			{
				//printf("調整前 weight[%d][%d] = %.9f\n", i, sy, weight[i][sy]);
				//printf("                      %.9f + 0.25 * (%.9f * %.9f * %.9f)\n", weight[i][sy], LEARNING_RATE, err[i][k], inputLayerValue[j]);
				weight[i][sy] = weight[i][sy] + 0.25 * (LEARNING_RATE * err[i][k] * inputLayerValue[j]);
				//printf("調整後 weight[%d][%d] = %.9f\n\n", i, sy, weight[i][sy]);
				sy++;
			}
			else if (i == layer - 1)//目前用不到
			for (int j = 0; j < eachLayerNeuron[i - 1]; j++)
			for (int k = 0; k < eachLayerNeuron[i]; k++)
			{
				printf("調整前 weight[%d][%d] = %.9f\n", i, sy, weight[i][sy]);
				printf("%.9f + 0.25 * (%.9f * %.9f * %.9f)\n", weight[i][sy], LEARNING_RATE, err[i][k], value[i - 1][k]);
				weight[i][sy] = weight[i][sy] + 0.25 * (LEARNING_RATE * err[i][k] * value[i - 1][k]);
				printf("調整後 weight[%d][%d] = %.9f\n", i, sy, weight[i][sy]);
				sy++;
			}
			else //兩層以上
			for (int j = 0; j < eachLayerNeuron[i + 1]; j++)
			for (int k = 0; k < eachLayerNeuron[i]; k++)
			{
				weight[i][sy] = weight[i][sy] + 0.25 * (LEARNING_RATE * err[i][k] * value[i][k]);
				sy++;
			}
			for (int j = 0; j < eachLayerNeuron[i]; j++) bias[i][j] = bias[i][j] + (LEARNING_RATE * err[i][j]);
		}
	}
};

int _tmain(int argc, _TCHAR* argv[])
{
	float insert[17];
	float LEARNING_RATE = 1.0, result[1];
	FILE *f;
	f = fopen("megData5.txt", "r");

	int layer = 1, neuron[1];
	BackPropagation bp(1);
	for (int i = 0; i < layer; i++)
	{
		cout << "Enter Layer " << (i + 1) << " Neuron: ";           //Input hiddenLayer and OutputLayer Neuron
		cin >> neuron[i];
	}
	bp.initialNeuron(17, neuron);

	for (int i = 0; i < 275; i++)
	{
		insert[0] = 1;
		fscanf(f, "%f", &result[0]);
		fscanf(f, "%f", &insert[1]);
		fscanf(f, "%f", &insert[2]);
		fscanf(f, "%f", &insert[3]);
		fscanf(f, "%f", &insert[4]);
		fscanf(f, "%f", &insert[5]);
		fscanf(f, "%f", &insert[6]);
		fscanf(f, "%f", &insert[7]);
		fscanf(f, "%f", &insert[8]);
		fscanf(f, "%f", &insert[9]);
		fscanf(f, "%f", &insert[10]);
		fscanf(f, "%f", &insert[11]);
		fscanf(f, "%f", &insert[12]);
		fscanf(f, "%f", &insert[13]);
		fscanf(f, "%f", &insert[14]);
		fscanf(f, "%f", &insert[15]);
		fscanf(f, "%f", &insert[16]);
		for (int j = 1; j < 17; j++)
		{
			if (insert[j] == 1 && j != 1 && j != 8) insert[j] = 1;
			else if(insert[j]==0 && j!=1 && j!=8) insert[j] = -1;
		}
		bp.inputValue(insert);
		bp.inputTrueValue(1, result);
		bp.caculation();
		bp.adjustWeightBias(LEARNING_RATE);
	}

	for (int i = 0; i < 275; i++)
	{
		insert[0] = 1;
		scanf("%f", &insert[1]);
		scanf("%f", &insert[2]);
		scanf("%f", &insert[3]);
		scanf("%f", &insert[4]);
		scanf("%f", &insert[5]);
		scanf("%f", &insert[6]);
		scanf("%f", &insert[7]);
		scanf("%f", &insert[8]);
		scanf("%f", &insert[9]);
		scanf("%f", &insert[10]);
		scanf("%f", &insert[11]);
		scanf("%f", &insert[12]);
		scanf("%f", &insert[13]);
		scanf("%f", &insert[14]);
		scanf("%f", &insert[15]);
		scanf("%f", &insert[16]);
		for (int j = 1; j < 17; j++)
		{
			if (insert[j] == 1 && j != 1 && j != 8) insert[j] = 10;
			else if (insert[j] == 0 && j != 1 && j != 8) insert[j] = -10;
		}
		bp.inputValue(insert);
		bp.inputTrueValue(1, result);
		bp.caculation();
	}
	/*int n = 0;
	for (int i = 0; i < 575; i++)
	{
	insert[0] = 1;
	insert[1] = v[n++];
	insert[2] = v[n++];
	insert[3] = v[n++];
	insert[4] = v[n++];
	insert[5] = v[n++];
	insert[6] = v[n++];
	insert[7] = v[n++];
	result[0] = v[n++];
	bp.inputValue(insert);
	bp.inputTrueValue(1, result);
	bp.caculation();
	bp.adjustWeightBias(LEARNING_RATE);
	}

	for (int i = 0; i < 5; i++)
	{
	insert[0] = 1;
	cout << "Age = ";
	cin >> insert[1];
	cout << "Beck = ";
	cin >> insert[2];
	cout << "IV Drug Use History = ";
	cin >> insert[3];
	cout << "Number of Prior Drug = ";
	cin >> insert[4];
	cout << "Subject's Race = ";
	cin >> insert[5];
	cout << "Treatment Randomization = ";
	cin >> insert[6];
	cout << "Treatment Site = ";
	cin >> insert[7];
	bp.inputValue(insert);
	bp.caculation();
	}*/
	return 0;
}


