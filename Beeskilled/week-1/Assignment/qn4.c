#include<stdio.h>
int list[] = {1,2,3,4,5};
int n=5;
void reverse(){
    printf("Original list is:\n");
    for(int i=0;i<n;i++){
        printf("%d ",list[i]);
    }
    int i=0,j=n-1,t;
    while(i<=j){
        t=list[i];
        list[i]=list[j];
        list[j]=t;
        i++;
        j--;
    }
    printf("\nReversed list is:\n");
    for(i=0;i<n;i++)
     printf("%d ",list[i]);
}
void main(){
    reverse();
    return;
}