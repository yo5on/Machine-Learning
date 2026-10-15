#include <stdio.h>
int list[] = {1, 2, 3, 4, 5};
int n=5;
void binary_search(int x){
    int f=0,l=n-1,m;
    while(f<=l){
        m=(f+l)/2;
        if(x>list[m])
        f=m+1;
        if(x<list[m])
        l=m-1;
        if(x==list[m]){
            printf("found at index:%d",m);
            return;
        }
    }
    printf("not found");
}
void main(){
    int x;
    printf("Enter the number to be searched:");
    scanf("%d",&x);
    binary_search(x);
}