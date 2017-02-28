#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THNN.h"
#else

TH_API void THNN_(Abs_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output);           // [OUT] Abs output
TH_API void THNN_(Abs_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. output
          THTensor *gradInput);        // [OUT] gradient w.r.t. input

TH_API void THNN_(AbsCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // tensor with target values
          THTensor *output,            // [OUT] a one-element tensor with loss
          long sizeAverage);           // if true, the loss will be divided by batch size
TH_API void THNN_(AbsCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // tensor with target values
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          long sizeAverage);           // if true, the gradient will be normalized by batch size

TH_API void THNN_(BCECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          long sizeAverage,
          THTensor *weights);          // [OPTIONAL]
TH_API void THNN_(BCECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          long sizeAverage,
          THTensor *weights);          // [OPTIONAL]

TH_API void THNN_(ClassNLLCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor (1D/2D)
          THIndexTensor *target,       // tensor containing indexes of target classes
          THTensor *output,            // [OUT] a one-element tensor with loss
          long sizeAverage,            // if true, the loss will be normalized by batch size and class weights
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight);     // [BUFFER]
TH_API void THNN_(ClassNLLCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor (1D/2D)
          THIndexTensor *target,       // tensor containing indexes of target classes
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          long sizeAverage,            // if true, the loss will be normalized by batch size and class weights
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight);     // [BUFFER]

TH_API void THNN_(SpatialClassNLLCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor (4D)
          THIndexTensor *target,       // tensor containing indexes of target classes (3D)
          THTensor *output,            // [OUT] a one-element tensor with loss
          long sizeAverage,            // if true, the loss will be normalized by batch size and class weights
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight);     // [BUFFER]
TH_API void THNN_(SpatialClassNLLCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor (4D)
          THIndexTensor *target,       // tensor containing indexes of target classes (3D)
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          long sizeAverage,            // if true, the loss will be normalized by batch size and class weights
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight);     // [BUFFER]

TH_API void THNN_(ELU_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] ELU output
          accreal alpha,               // an ELU parameter (as in paper)
          long inplace);               // if true, modifies gradOutput and sets gradInput onto it (no additional memory is allocated)
TH_API void THNN_(ELU_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          THTensor *output,            // output from a forward pass
          accreal alpha,               // an ELU parameter (as in paper)
          long inplace);               // if true, modifies gradOutput and sets gradInput onto it (no additional memory is allocated)

TH_API void THNN_(DistKLDivCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor
          THTensor *output,            // [OUT] a one-element tensor containing the loss
          long sizeAverage);           // if true, the loss will be normalized **by total number of elements**
TH_API void THNN_(DistKLDivCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          long sizeAverage);           // if true, the loss will be normalized **by total number of elements**

TH_API void THNN_(GatedLinear_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] output tensor, half size of input along dimension dim
          long dim);                    // dimension for halving operation
TH_API void THNN_(GatedLinear_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t input
          long dim);                    // dimension for halving operation

// HardShink outputs 0 on interval of (-lambda; lambda) or original value otherwise.
TH_API void THNN_(HardShrink_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] output tensor
          accreal lambda);             // HardShrink parameter
TH_API void THNN_(HardShrink_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          accreal lambda);             // HardShrink parameter

// HardTanh clamps the values to the interval [min_val; max_val].
TH_API void THNN_(HardTanh_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] output tensor
          accreal min_val,             // lower threshold
          accreal max_val,             // upper threshold
          long inplace);
TH_API void THNN_(HardTanh_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. the input
          accreal min_val,             // lower threshold
          accreal max_val,             // upper threshold
          long inplace);

TH_API void THNN_(L1Cost_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output);           // [OUT] output tensor
TH_API void THNN_(L1Cost_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // [OPTIONAL] gradient w.r.t module's output
          THTensor *gradInput);        // [OUT] gradient w.r.t the input

TH_API void THNN_(LeakyReLU_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // [MODIFIED] input tensor
          THTensor *output,            // [OUT] output tensor
          accreal negval,              // negative part slope
          long inplace);               // if true, modifies the input tensor and sets the output tensor on it (no additional memory is allocated)
TH_API void THNN_(LeakyReLU_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // [MODIFIED] gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. the input
          accreal negval,              // negative part slope
          long inplace);               // if true, modifies gradOutput and sets gradInput onto it (no additional memory is allocated)

TH_API void THNN_(LogSigmoid_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // output tensor
          THTensor *buffer);           // [BUFFER]
TH_API void THNN_(LogSigmoid_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          THTensor *buffer);           // [BUFFER]

TH_API void THNN_(LogSoftMax_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output);           // [OUT] output tensor
TH_API void THNN_(LogSoftMax_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          THTensor *output);           // module's output

TH_API void THNN_(LookupTable_accGradParameters)(
          THNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THIntegerTensor *count,
          THTensor *sorted,            // [OPTIONAL]
          THIndexTensor *indices,      // [OPTIONAL]
          long scaleGradByFreq,
          long paddingValue,
          accreal scale);

TH_API void THNN_(LookupTable_renorm)(
          THNNState *state,            // library's state
          THIndexTensor *idx,          // vector containing row indices (modified in function)
          THTensor *weight,            // 2D tensor whose rows will be renormalized
          accreal maxNorm,             // maximum norm
          accreal normType);           // the norm type (e.g., normType=2, then it's 2-norm)

TH_API void THNN_(MarginCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor (should contain only 1s and -1s)
          THTensor *output,            // [OUT] a one-element tensor containing the loss
          long sizeAverage,            // if true, the loss is normalized by **total number of elements**
          accreal margin);             // a margin that is required for the loss to be 0

TH_API void THNN_(MarginCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor (should contin only 1s and -1s)
          THTensor *gradInput,         // [OUT] gradient w.r.t. module's input
          long sizeAverage,            // if true, the gradient is normalized by **total number of elements**
          accreal margin);             // a margin that is required for the loss to be 0

TH_API void THNN_(SoftMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          long sizeAverage);

TH_API void THNN_(SoftMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          long sizeAverage);

TH_API void THNN_(MSECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          long sizeAverage);
TH_API void THNN_(MSECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          long sizeAverage);

TH_API void THNN_(MultiLabelMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          THTensor *isTarget,
          long sizeAverage);
TH_API void THNN_(MultiLabelMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          THTensor *isTarget,
          long sizeAverage);

TH_API void THNN_(MultiMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          long sizeAverage,
          long p,
          THTensor* weights,      // [OPTIONAL]
          accreal margin);
TH_API void THNN_(MultiMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          long sizeAverage,
          long p,
          THTensor *weights,      // [OPTIONAL]
          accreal margin);

TH_API void THNN_(PReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THIndex_t nOutputPlane);
TH_API void THNN_(PReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THIndex_t nOutputPlane);
TH_API void THNN_(PReLU_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradWeight,
          THTensor *gradWeightBuf,
          THTensor *gradWeightBuf2,
          THIndex_t nOutputPlane,
          accreal scale);

TH_API void THNN_(Linear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *addBuffer);
TH_API void THNN_(Linear_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight);
TH_API void THNN_(Linear_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *addBuffer,
          accreal scale);

TH_API void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          accreal lower,
          accreal upper,
          long train,
          long inplace,
          THGenerator *generator);
TH_API void THNN_(RReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          accreal lower,
          accreal upper,
          long train,
          long inplace);

TH_API void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(SmoothL1Criterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          long sizeAverage);
TH_API void THNN_(SmoothL1Criterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          long sizeAverage);

TH_API void THNN_(SoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(SoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(SoftPlus_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal beta,
          accreal threshold);
TH_API void THNN_(SoftPlus_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          accreal beta,
          accreal threshold);

TH_API void THNN_(SoftShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal lambda);
TH_API void THNN_(SoftShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal lambda);

TH_API void THNN_(SparseLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias);
TH_API void THNN_(SparseLinear_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(SparseLinear_zeroGradParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput);
TH_API void THNN_(SparseLinear_updateParameters)(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          accreal learningRate);
TH_API void THNN_(SparseLinear_legacyUpdateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias);
TH_API void THNN_(SparseLinear_legacyAccGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(SparseLinear_legacyZeroGradParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput);
TH_API void THNN_(SparseLinear_legacyUpdateParameters)(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          accreal learningRate);

TH_API void THNN_(Sqrt_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal eps);
TH_API void THNN_(Sqrt_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(Square_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Square_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(Tanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Tanh_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal threshold,
          accreal val,
          long inplace);
TH_API void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal threshold,
          accreal val,
          long inplace);

TH_API void THNN_(TemporalConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          long kW, long dW,
          long inputFrameSize,
          long outputFrameSize);
TH_API void THNN_(TemporalConvolution_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          long kW, long dW);
TH_API void THNN_(TemporalConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          long kW, long dW,
          accreal scale);
TH_API void THNN_(TemporalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long kW, long dW);
TH_API void THNN_(TemporalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          long kW, long dW);
TH_API void THNN_(TemporalSubSampling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          long kW, long dW,
          long inputFrameSize);
TH_API void THNN_(TemporalSubSampling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          long kW, long dW);
TH_API void THNN_(TemporalSubSampling_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          long kW, long dW,
          accreal scale);

TH_API void THNN_(TemporalRowConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          long kW,
          long dW,
          long padW,
          long featFirst);
TH_API void THNN_(TemporalRowConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          long kW,
          long dW,
          long padW,
          long featFirst);
TH_API void THNN_(TemporalRowConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          long kW,
          long dW,
          long padW,
          long featFirst,
          accreal scale);

TH_API void THNN_(BatchNormalization_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,       // [OPTIONAL]
          THTensor *bias,         // [OPTIONAL]
          THTensor *running_mean,
          THTensor *running_var,
          THTensor *save_mean,
          THTensor *save_std,
          long train,
          double momentum,
          double eps);
TH_API void THNN_(BatchNormalization_backward)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,    // [OPTIONAL]
          THTensor *gradWeight,   // [OPTIONAL]
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *weight,       // [OPTIONAL]
          THTensor *running_mean,
          THTensor *running_var,
          THTensor *save_mean,
          THTensor *save_std,
          long train,
          double scale,
          double eps);

TH_API void THNN_(SpatialConvolutionMap_updateOutput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *output,       // [OUT] convolution output
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          long nInputPlane,        // number of input planes
          long nOutputPlane,       // number of output planes
          long dW, long dH);        // stride
TH_API void THNN_(SpatialConvolutionMap_updateGradInput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradInput,    // [OUT] gradient w.r.t. input
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          long nInputPlane,        // number of input planes
          long nOutputPlane,       // number of output planes
          long dW, long dH);        // stride
TH_API void THNN_(SpatialConvolutionMap_accGradParameters)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradWeight,   // 3D gradWeight tensor (connTable:size(1) x kH x kW)
          THTensor *gradBias,     // 1D gradBias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          long nInputPlane,        // number of input planes
          long nOutputPlane,       // number of output planes
          long dW, long dH,         // stride
          accreal scale);         // scaling factor

TH_API void THNN_(SpatialConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH);
TH_API void THNN_(SpatialConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH);
TH_API void THNN_(SpatialConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          accreal scale);

TH_API void THNN_(SpatialConvolutionLocal_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
TH_API void THNN_(SpatialConvolutionLocal_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
TH_API void THNN_(SpatialConvolutionLocal_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight,
          accreal scale);

TH_API void THNN_(SpatialAdaptiveMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long owidth, long oheight);
TH_API void THNN_(SpatialAdaptiveMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices);

TH_API void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long owidth, long oheight);
TH_API void THNN_(SpatialAdaptiveAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(SpatialAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long ceil_mode,
          long count_include_pad);
TH_API void THNN_(SpatialAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long ceil_mode,
          long count_include_pad);

TH_API void THNN_(SpatialFractionalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long outputW, long outputH,
          long poolSizeW, long poolSizeH,
          THIndexTensor *indices,
          THTensor *randomSamples);
TH_API void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          long outputW, long outputH,
          long poolSizeW, long poolSizeH,
          THIndexTensor *indices);

TH_API void THNN_(SpatialFullConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long adjW, long adjH);
TH_API void THNN_(SpatialFullConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradColumns,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long adjW, long adjH);
TH_API void THNN_(SpatialFullConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long adjW, long adjH,
          accreal scale);

TH_API void THNN_(SpatialFullConvolutionMap_updateOutput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *output,       // [OUT] convolution output
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          long nInputPlane,        // number of input planes
          long nOutputPlane,       // number of output planes
          long dW, long dH);        // stride
TH_API void THNN_(SpatialFullConvolutionMap_updateGradInput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradInput,    // [OUT] gradient w.r.t. input
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          long nInputPlane,        // number of input planes
          long nOutputPlane,       // number of output planes
          long dW, long dH);        // stride
TH_API void THNN_(SpatialFullConvolutionMap_accGradParameters)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradWeight,   // 3D gradWeight tensor (connTable:size(1) x kH x kW)
          THTensor *gradBias,     // 1D gradBias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          long nInputPlane,        // number of input planes
          long nOutputPlane,       // number of output planes
          long dW, long dH,         // stride
          accreal scale);         // scaling factor

TH_API void THNN_(SpatialDilatedConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long dilationW, long dilationH);

TH_API void THNN_(SpatialDilatedConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradColumns,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long dilationW, long dilationH);

TH_API void THNN_(SpatialDilatedConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long dilationW, long dilationH,
          accreal scale);

TH_API void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long ceil_mode);
TH_API void THNN_(SpatialMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long ceil_mode);

TH_API void THNN_(SpatialDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long dilationW, long dilationH,
          long ceil_mode);
TH_API void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long dilationW, long dilationH,
          long ceil_mode);

TH_API void THNN_(SpatialMaxUnpooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long owidth, long oheight);
TH_API void THNN_(SpatialMaxUnpooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          long owidth, long oheight);

TH_API void THNN_(SpatialSubSampling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          long kW, long kH,
          long dW, long dH);
TH_API void THNN_(SpatialSubSampling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          long kW, long kH,
          long dW, long dH);
TH_API void THNN_(SpatialSubSampling_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          long kW, long kH,
          long dW, long dH,
          accreal scale);

TH_API void THNN_(SpatialUpSamplingNearest_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long scale_factor);
TH_API void THNN_(SpatialUpSamplingNearest_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          long scale_factor);

TH_API void THNN_(SpatialUpSamplingBilinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
	  long outputHeight,
          long outputWidth);
TH_API void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          long nbatch,
          long nchannels,
          long inputHeight,
          long inputWidth,
          long outputHeight,
          long outputWidth);

TH_API void THNN_(unfolded_acc)(
          THTensor *finput,
          THTensor *input,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long nInputPlane,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
TH_API void THNN_(unfolded_copy)(
          THTensor *finput,
          THTensor *input,
          long kW, long kH,
          long dW, long dH,
          long padW, long padH,
          long nInputPlane,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);

TH_API void THNN_(VolumetricAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long kT, long kW, long kH,
          long dT, long dW, long dH);
TH_API void THNN_(VolumetricAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          long kT, long kW, long kH,
          long dT, long dW, long dH);

TH_API void THNN_(VolumetricConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,           // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          long dT, long dW, long dH,
          long pT, long pW, long pH);
TH_API void THNN_(VolumetricConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          long dT, long dW, long dH,
          long pT, long pW, long pH);
TH_API void THNN_(VolumetricConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,       // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          long dT, long dW, long dH,
          long pT, long pW, long pH,
          accreal scale);

TH_API void THNN_(VolumetricConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,           // [OPTIONAL]
          THTensor *finput,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long pT, long pW, long pH);
TH_API void THNN_(VolumetricConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long pT, long pW, long pH);
TH_API void THNN_(VolumetricConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,       // [OPTIONAL]
          THTensor *finput,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long pT, long pW, long pH,
          accreal scale);

TH_API void THNN_(VolumetricFractionalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long outputT, long outputW, long outputH,
          long poolSizeT, long poolSizeW, long poolSizeH,
          THIndexTensor *indices,
          THTensor *randomSamples);
TH_API void THNN_(VolumetricFractionalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          long outputT, long outputW, long outputH,
          long poolSizeT, long poolSizeW, long poolSizeH,
          THIndexTensor *indices);

TH_API void THNN_(VolumetricFullConvolution_updateOutput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *output,         // [OUT] volumetric convolution output
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *bias,           // [OPTIONAL] gradBias tensor (nOutputPlane)
          THTensor *finput,         // [OUT] internal columns buffer
          THTensor *fgradInput,     // [OUT] internal ones buffer
          long dT, long dW, long dH,   // stride of the convolution
          long pT, long pW, long pH,   // padding
          long aT, long aW, long aH);  // extra output adjustment
TH_API void THNN_(VolumetricFullConvolution_updateGradInput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradInput,      // [OUT] gradient w.r.t. input
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          long dT, long dW, long dH,   // stride
          long pT, long pW, long pH,   // padding
          long aT, long aW, long aH);  // extra output adjustment
TH_API void THNN_(VolumetricFullConvolution_accGradParameters)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradWeight,     // gradWeight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *gradBias,       // [OPTIONAL] gradBias tensor (nOutputPlane)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          long dT, long dW, long dH,   // stride
          long pT, long pW, long pH,   // padding
          long aT, long aW, long aH,   // extra output adjustment
          accreal scale);           // scaling factor

TH_API void THNN_(VolumetricDilatedConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,           // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long padT, long padW, long padH,
          long dilationT, long dilationW, long dilationH);

TH_API void THNN_(VolumetricDilatedConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradColumns,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long padT, long padW, long padH,
          long dilationT, long dilationW, long dilationH);

TH_API void THNN_(VolumetricDilatedConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,       // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long padT, long padW, long padH,
          long dilationT, long dilationW, long dilationH,
          accreal scale);

TH_API void THNN_(VolumetricMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long pT, long pW, long pH,
          long ceilMode);
TH_API void THNN_(VolumetricMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long pT, long pW, long pH,
          long ceilMode);

TH_API void THNN_(VolumetricDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long pT, long pW, long pH,
          long dilationT, long dilationW, long dilationH,
          long ceilMode);
TH_API void THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          long kT, long kW, long kH,
          long dT, long dW, long dH,
          long pT, long pW, long pH,
          long dilationT, long dilationW, long dilationH,
          long ceilMode);

TH_API void THNN_(VolumetricMaxUnpooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long oT, long oW, long oH,
          long dT, long dW, long dH,
          long pT, long pW, long pH);
TH_API void THNN_(VolumetricMaxUnpooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          long oT, long oW, long oH,
          long dT, long dW, long dH,
          long pT, long pW, long pH);

TH_API void THNN_(SpatialReflectionPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long pad_l, long pad_r,
          long pad_t, long pad_b);

TH_API void THNN_(SpatialReflectionPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          long pad_l, long pad_r,
          long pad_t, long pad_b);

TH_API void THNN_(SpatialReplicationPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long pad_l, long pad_r,
          long pad_t, long pad_b);

TH_API void THNN_(SpatialReplicationPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          long pad_l, long pad_r,
          long pad_t, long pad_b);

TH_API void THNN_(VolumetricReplicationPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          long pleft, long pright,
          long ptop, long pbottom,
          long pfront, long pback);

TH_API void THNN_(VolumetricReplicationPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          long pleft, long pright,
          long ptop, long pbottom,
          long pfront, long pback);
#endif
