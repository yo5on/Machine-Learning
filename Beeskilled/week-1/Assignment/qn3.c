#include<stdio.h>
int list[] = {2,4,6,3,1,5,9,7,8};
int n=9;
void bubble_sort(){
    printf("Original list is:\n");
    for(int i=0;i<n;i++){
        printf("%d ",list[i]);
    }
    int i,j,t;
    for(i=0;i<n-1;i++){
        for(j=0;j<n-i-1;j++){
            if(list[j]>list[j+1]){
                t=list[j];
                list[j]=list[j+1];
                list[j+1]=t;
            }
        }
    }
    printf("\nSorted list is:\n");
    for(i=0;i<n;i++){
        printf("%d ",list[i]);
    }
}void main(){
    bubble_sort();
}