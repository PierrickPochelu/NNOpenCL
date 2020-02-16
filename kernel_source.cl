__kernel void matrix_multiply(
__global float * matrixA,
__global float * matrixB,
__global float * output)
{
    size_t id = get_global_id(0);
    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id+2] = matrixA[id+2] * matrixB[id+2];
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1];
    }
}

__kernel void matrix_add(
__global float * matrixA,
__global float * matrixB,
__global float * output)
{
    size_t id = get_global_id(0);
    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id+2] = matrixA[id+2] + matrixB[id+2];
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1];
    }
}

__kernel void matrix_minus(
__global float * matrixA,
__global float * matrixB,
__global float * output)
{
    size_t id = get_global_id(0);
    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id+2] = matrixA[id+2] - matrixB[id+2];
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1];
    }
}

__kernel void matrix_zero(
__global float * matrixA,
__global float * output)
{
    size_t id = get_global_id(0);
    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id+2] = 0;
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1];
    }
}

__kernel void matrix_dot(
__global float * matrixA,
__global float * matrixB,
__global float * output)
{
    size_t id = get_global_id(0);
    
    uint nb_add=(uint) matrixA[1];/*matrixA[1]==matrixB[0]==nb addition by element*/
    uint nb_row_out=(uint) matrixA[0];
    uint nb_col_out=(uint) matrixB[1];

    //coord of the element computed in output
    uint row_id_out=id / nb_col_out;
    uint col_id_out=id % nb_col_out;

    //computing
    uint coord_A=0;
    uint coord_B=0;
    if(id<nb_row_out*nb_col_out){//only id=0 and id=1
        output[id + 2]=0;
        for(uint i=0; i<nb_add; i++){
            coord_A=row_id_out * (uint) matrixA[1] +     i;
            coord_B=     i     * (uint) matrixB[1] +   col_id_out;
            output[id + 2] += matrixA[coord_A+2] * matrixB[coord_B+2];
        }
    }

    //nb row/col
    if(id==0){
        output[0]=(float)nb_row_out;
        output[1]=(float)nb_col_out;
    }
}

__kernel void matrix_dot_scalar(
__global float * matrixA,
__global float * scalar,
__global float * output)
{
    size_t id = get_global_id(0);
    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id + 2] = matrixA[id+2] * scalar[2];
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1];
    }
}

__kernel void matrix_add_scalar(
__global float * matrixA,
__global float *  scalar,
__global float * output)
{
    size_t id = get_global_id(0);
    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id + 2] = matrixA[id+2] + scalar[2];
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1];
    }
}

__kernel void matrix_transpose(
__global float * matrixA,
__global float * output)
{
    size_t id = get_global_id(0);
    uint w=(uint) matrixA[0];
    uint h=(uint) matrixA[1];
    uint row_id=id / w;
    uint col_id=id % w;
    uint coord_old=0;
    uint coord_new=0;
    coord_old=row_id*w+col_id;
    coord_new=col_id*h+row_id;

    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[coord_new + 2] = matrixA[coord_old+2];
    }
    if(id==0){
        //invert w and h
        output[0]=(float)h;
        output[1]=(float)w;
    }
}


// Warning : work only with vector
__kernel void matrix_addbias(
__global float * matrixA,
__global float * output){
    size_t id=get_global_id(0);

    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id+3]=matrixA[id+2];
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1]+1;
        output[2]=1;
    }
}


// Warning : work only with vector
__kernel void matrix_rmbias(
__global float * matrixA,
__global float * output){
    size_t id=get_global_id(0);

    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id+2]=matrixA[id+3];
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1]-1;
    }
}


__kernel void matrix_copy(
__global float * matrixA,
__global float * output){
    size_t id=get_global_id(0);

    if(id<(uint) (matrixA[0]*matrixA[1])){
        output[id+2]=matrixA[id+2];
    }
    if(id==0){
        output[0]=matrixA[0];
        output[1]=matrixA[1];
    }
}

//////////////////////////////////////////////////////////

__kernel void mlp_activ(
__global float * matrixA,
__global float * output){
    size_t id = get_global_id(0);
    if(id<(uint) (matrixA[0]*matrixA[1])){

        output[id+2] = 1/(1+exp(-1*matrixA[id+2]));

    }
    if(id==0){
        output[0] = matrixA[0];
        output[1] = matrixA[1];
    }
}


//sigmoid = 1/(1+exp(-x))
//df = f * (1 - f)

__kernel void mlp_dactiv(
__global float * matrixA,
__global float * output){
    size_t id = get_global_id(0);
    if(id<(uint) (matrixA[0]*matrixA[1])){
            output[id+2] = matrixA[id+2] * (1 - matrixA[id+2]);
    }
    if(id==0){
        output[0] = matrixA[0];
        output[1] = matrixA[1];
    }

}


kernel void mlp_loss(
__global float * predicted,
__global float * groundtrue,
__global float * loss){
    size_t id = get_global_id(0);
    uint i=2;
    if(id==0){
        loss[0] = predicted[0];
        loss[1] = predicted[1];
        loss[i] = (predicted[i]-groundtrue[i])*(predicted[i]-groundtrue[i]);
    }
}

__kernel void mlp_dloss(
__global float * predicted,
__global float * groundtrue,
__global float * loss){
    size_t id = get_global_id(0);
    uint i=2;
    if(id==0){
        loss[0] = predicted[0];
        loss[1] = predicted[1];
        loss[i] = 2*(predicted[i]-groundtrue[i]);
    }
}



__kernel void mlp_inference(
__global float * x,
__global float * label,
__global float DATA[100][100], //DATA[0] is variable 1, DATA[1] is variable 2 etc...
__global float * input_w1,
__global float * input_w2){
    size_t id=get_global_id(0);

    matrix_addbias(x,DATA[0]);
    matrix_dot(DATA[0],input_w1,DATA[1]);
    mlp_activ(DATA[1],DATA[2]);

    matrix_addbias(DATA[2],DATA[3]);
    matrix_dot(DATA[3],input_w2,DATA[4]);
    mlp_activ(DATA[4],DATA[5]);

    mlp_loss(DATA[5],label,DATA[6]);
}


__kernel void mlp_training(
__global float list_x[100][100],//headers contains stats on data
__global float list_y[100][100],
__global float DATA[100][100], //DATA[0] is variable 1, DATA[1] is variable 2 etc...
__global float * w1,
__global float * w2,
__global float params[4]){ // nb_data, iteration, batch_size, lr
    size_t id=get_global_id(0);

    __global static uint nb_data;
    nb_data = (uint) params[0];
    __global static uint iteration;
    iteration = (uint)params[1];
    __global static uint batch_size;
    batch_size = (uint)params[2];
    __global static float lr[3];
    lr[0]=1;lr[1]=1;lr[2]=params[3];
    __global static float two[3]={1,1,2};

    matrix_zero(w1,DATA[15]);//w1_grad init. 0
    matrix_zero(w2,DATA[21]);//w2_grad init. 0



    for(uint j=0;j<iteration;j++){
        for(uint i=0;i<nb_data;i++){
            //INFERENCE
            matrix_addbias(list_x[i],DATA[0]);
            matrix_dot(DATA[0],w1,DATA[1]);
            mlp_activ(DATA[1],DATA[2]);
            matrix_addbias(DATA[2],DATA[3]);
            matrix_dot(DATA[3],w2,DATA[4]);//DATA[4]=a3
            mlp_activ(DATA[4],DATA[5]); 
            mlp_loss(DATA[5],list_y[i],DATA[6]);

            //COMPUTE DELTA
            mlp_dloss(DATA[5],list_y[i],DATA[7]);//DATA[7]=w2_delta
            matrix_dot_scalar(w2,DATA[7],DATA[8]);
            mlp_dactiv(DATA[4],DATA[9]);
            matrix_multiply(DATA[8],DATA[9],DATA[10]);
            matrix_transpose(DATA[10],DATA[11]);
            matrix_rmbias(DATA[11],DATA[11]);//DATA[11]=w1_delta

            //delta_a2b=multiply(dot(w2,w2_delta),dactive(a3))
            //w1_delta=rmbias(delta_a2b)

            //COMPUTE W1_GRADS
            matrix_transpose(DATA[0],DATA[12]);//DATA[12]=transpose(a1b)
            matrix_dot_scalar(DATA[12],two,DATA[13]);//DATA[13]=2*transpose_vector(a1b)
            matrix_dot(DATA[13],DATA[11],DATA[14]);//DATA[14]=w1_delta*2*transpose_vector(a1b)
            matrix_add_scalar(DATA[15],DATA[14],DATA[15]);//DATA[15]=update DATA[14]=w1_grad_cumul

            matrix_transpose(DATA[3],DATA[17]);//DATA[17]=transpose(a2b)
            matrix_dot_scalar(DATA[17],two,DATA[18]);//DATA[18]=2*transpose_vector(a2b)
            matrix_dot(DATA[18],DATA[7],DATA[19]);//DATA[19]=w2_delta*2*transpose_vector(a2b)
            matrix_add_scalar(DATA[21],DATA[19],DATA[21]);//DATA[21]=update DATA[19]=w2_grad_cumul

            //w1_grad=w1_grad+w1_delta*2*transpose_vector(a1b) # w1_grad must be (5,3)=(5,3)+(1,5)*(3,1)
            //w2_grad=w2_grad+w2_delta*2*transpose_vector(a2b) # w2_grad must be 6,1 = (6,1)+(6,1)

            //BATCH UPDATE
            if((i % batch_size == 0 && i != 0) || i == nb_data-1){
                matrix_dot_scalar(DATA[15],lr,DATA[22]);
                matrix_minus(w1,DATA[22],w1);//w1=w1-w1_grad*lr

                matrix_dot_scalar(DATA[21],lr,DATA[23]);
                matrix_minus(w2,DATA[23],w2);//w2=w2-w2_grad*lr


                matrix_zero(w1,DATA[15]);//w1_grad init. 0
                matrix_zero(w2,DATA[21]);//w2_grad init. 0
            }

        }
    }
}
