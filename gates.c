#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>

typedef  float Sample[2];


typedef struct{
    int len,dimensions;
    Sample* sample;
} Dataset;


typedef struct {
    float b;
    float* weights;
    Dataset* dataset;
} Perceptron;



Sample train_data[]={
    {0,2},
    {1,4},
    {2,6},
    {3,8},
    {4,10}
};

// Sample or_data[]={
//     {0,0,0},
//     {0,1,1},
//     {1,0,1},
//     {1,1,1}
// };

// Sample and_data[]={
//     {0,0,0},
//     {0,1,0},
//     {1,0,0},
//     {1,1,1}
// };



// Sample nor_data[]={
//     {0,0,1},
//     {0,1,0},
//     {1,0,0},
//     {1,1,0}
// };






float sigmoid(float x)
{
    return 1/(1+exp(-x));
}


float forward(Perceptron *p,float *inputs)
{
    int dim=p->dataset->dimensions;
    float res=p->b;
    for(int j=0;j<dim;j++)
    {
        res+=inputs[j]*p->weights[j];
    }
    return (res);
}

float cost(Perceptron *p)
{
    float result=0.0;
    int len=p->dataset->len;
    int dim=p->dataset->dimensions;
    for(int i=0;i<len;i++)
    {
        float d=p->dataset->sample[i][dim]-forward(p,p->dataset->sample[i]);
        result+=d*d;
    }
    return result/len;
}



void gradient_descent(Perceptron *p,float lr)
{
    float eps=1e-3;
    float c=cost(p);
    int size=p->dataset->dimensions;
    float* dw=malloc(sizeof(float)*size);
    for(int i=0;i<size;i++)
    {
        p->weights[i]+=eps;
        dw[i]=(cost(p)-c)/eps;
        p->weights[i]-=eps;
    }
    for(int i=0;i<size;i++)
    {
        p->weights[i]-=lr*dw[i];
    }
    p->b+=eps;
    float db=(cost(p)-c)/eps;
    p->b-=eps;
    p->b=p->b-lr*db;
    free(dw);
}



void print_model(Perceptron* p)
{
    int size=p->dataset->dimensions;
    for(int i=0;i<size;i++)
    {
        printf(" %f ",p->weights[i]);
    }
    printf(" %f\n",p->b);
}

void train(Perceptron* p,int iterations,float lr)
{
    
    for(int i=0;i<iterations;i++)
    {
        gradient_descent(p,lr);
        printf("%f\n",cost(p));
    }
}

float randfloat()
{
    return (float)rand()/(float)RAND_MAX;
}


Perceptron* init_perceptron(Dataset* ds)
{
    Perceptron* p = malloc(sizeof(Perceptron));
    int dim=ds->dimensions;
    p->weights=malloc(sizeof(float)*dim);
    for(int i=0;i<dim;i++)
    {
        p->weights[i]=randfloat();
    }
    p->b=randfloat();
    p->dataset=ds;
    return p;
}


int main()
{   
    srand(time(0));
    Dataset dataset={.len=sizeof(train_data)/sizeof(Sample),.sample=train_data,.dimensions=(sizeof(Sample)/sizeof(float))-1};
    Perceptron* p=init_perceptron(&dataset);
    float inputs[2];
    train(p,500*1000,0.001);
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<2;j++){
    //         inputs[0]=i;inputs[1]=j;
    //         printf("%d %d %f\n",i,j,forward(p,inputs));
    //     }
    // }
    float *f=malloc(sizeof(float));
    for(int i=0;i<10;i++)
    {
        *f=i;
        printf("%d %f\n",i,forward(p,f));
    }
    print_model(p);
}