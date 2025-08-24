#include<stdio.h>
#include<stdlib.h>
#include<time.h>

typedef float Sample[2];


Sample train_data[]={
    {0,2},
    {1,4},
    {2,6},
    {3,8},
    {4,10}
};

Sample* sample=train_data;


typedef struct {
    float w,b;
} LinearModel;

#define train_size 5

float forward(LinearModel l,float x1)
{
    return l.w*x1+l.b;
}

float cost(LinearModel p)
{
    
    float result=0.0;
    for(int i=0;i<train_size;i++)
    {
        float d=sample[i][1]-forward(p,sample[i][0]);
        result+=d*d;
    }
    return result/train_size;
}



LinearModel gradient(LinearModel p)
{
    float eps=1e-3;
    float c=cost(p);
    float dw=(cost((LinearModel){p.w+eps,p.b})-c)/eps;
    float db=(cost((LinearModel){p.w,p.b+eps})-c)/eps;
    return (LinearModel){dw,db};
}

LinearModel train(LinearModel p,int iterations,float lr)
{
    for(int i=0;i<iterations;i++)
    {
        LinearModel g=gradient(p);
        p.w-=(lr*g.w);
        p.b-=(lr*g.b);
        printf("%f\n",cost(p));
    }

    return p;
}


float randfloat()
{
    return (float)rand()/(float)RAND_MAX;
}

int main()
{
    srand(time(0));
    LinearModel l={randfloat(),randfloat()};
    l=train(l,5000,0.1);
    for(int i=0;i<10;i++)printf("%d %f\n",i,forward(l,i));
    printf("%f %f\n",l.w,l.b);
}
