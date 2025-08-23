#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>

typedef  float Sample[3];

typedef struct {
    float w1,w2,b;
} Perceptron;


Sample or_data[]={
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1}
};

Sample and_data[]={
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1}
};



Sample nor_data[]={
    {0,0,1},
    {0,1,0},
    {1,0,0},
    {1,1,0}
};



Sample* sample_data= and_data;

#define train_size 4



float sigmoid(float x)
{
    return 1/(1+exp(-x));
}


float forward(Perceptron p,float x1,float x2)
{
    return sigmoid(p.w1*x1 + p.w2*x2 +p.b);
}

float cost(Perceptron p)
{
    float result=0.0;
    for(int i=0;i<train_size;i++)
    {
        float d=sample_data[i][2]-forward(p,sample_data[i][0],sample_data[i][1]);
        result+=d*d;
    }
    return result/train_size;
}



Perceptron gradient(Perceptron p)
{
    float eps=1e-2;
    float c=cost(p);
    float dw1=(cost((Perceptron){p.w1+eps,p.w2,p.b})-c)/eps;
    float dw2=(cost((Perceptron){p.w1,p.w2+eps,p.b})-c)/eps;
    float db=(cost((Perceptron){p.w1,p.w2,p.b+eps})-c)/eps;
    return (Perceptron){dw1,dw2,db};
}

Perceptron train(Perceptron p,int iterations,float lr)
{
    printf("%f %f %f\n",p.w1,p.w2,p.b);
    for(int i=0;i<iterations;i++)
    {
        Perceptron g=gradient(p);
        p.w1-=(lr*g.w1);
        p.w2-=(lr*g.w2);
        p.b-=(lr*g.b);
        printf("%f\n",cost(p));
    }
    printf("%f %f %f\n",p.w1,p.w2,p.b);

    return p;
}


float randfloat()
{
    return (float)rand()/(float)RAND_MAX;
}

int main()
{   
    srand(time(0));
    Perceptron p={randfloat(),randfloat(),randfloat()};
    p=train(p,500*1000,0.01);
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++)
            printf("%d %d %f\n",i,j,forward(p,i,j));
    }
}