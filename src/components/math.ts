// The dot product of two arrays of the same length
export const dotProduct = (a: number[], b: number[]) => {
    if(a.length !== b.length) {
        throw new Error("Arrays must be of the same length");
    }
    let sum = 0;
    for(let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}