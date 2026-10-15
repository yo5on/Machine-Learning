#include <stdio.h>
int list[] = {1,2,3,4,5};
int n = 5;
void linear_search(int x){
    int i;
    for (i=0;i<n;i++)
    if(list[i]==x){
        printf("found at index:%d",i);
        return;
    }
    
    if(i==n)
    printf("not found");
}
void main(){
    int x;
    printf("Enter the number to search:");
    scanf("%d",&x);
    linear_search(x);
}