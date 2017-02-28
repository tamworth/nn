#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxPooling.c"
#else

void THNN_(VolumetricMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          long kT,
          long kW,
          long kH,
          long dT,
          long dW,
          long dH,
          long pT,
          long pW,
          long pH,
          long ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateOutput)(
          state, input, output, indices,
          kT, kW, kH, dT, dW, dH,
          pT, pW, pH, 1, 1, 1, ceilMode);
}

void THNN_(VolumetricMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          long kT,
          long kW,
          long kH,
          long dT,
          long dW,
          long dH,
          long pT,
          long pW,
          long pH,
          long ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
          state, input, gradOutput, gradInput, indices,
          kT, kW, kH, dT, dW, dH,
          pT, pW, pH, 1, 1, 1, ceilMode);
}

#endif
