//+------------------------------------------------------------------+
//|                                                   FFTUtility.mqh |
//|                        FFT Implementation for ATFNet Integration |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| FFT Utility Class                                                 |
//+------------------------------------------------------------------+
class CFFTUtility {
private:
    int fftSize;
    double m_real[];    // Real part array
    double m_imag[];    // Imaginary part array
    
public:
    CFFTUtility(int size) {
        // FFT works best with power of 2 sizes
        fftSize = NearestPowerOf2(size);
        ArrayResize(m_real, fftSize);
        ArrayResize(m_imag, fftSize);
    }
    
    // Find nearest power of 2
    int NearestPowerOf2(int size) {
        int power = 1;
        while(power < size) {
            power *= 2;
        }
        return power;
    }
    
    // Perform FFT on real data
    bool PerformFFT(const double& inReal[], double& outReal[], double& outImag[], bool inverse = false) {
        int n = ArraySize(inReal);
        if(n == 0) return false;
        
        // Resize output arrays
        ArrayResize(outReal, fftSize);
        ArrayResize(outImag, fftSize);
        ArrayInitialize(outReal, 0);
        ArrayInitialize(outImag, 0);
        
        // Copy input with padding
        for(int i = 0; i < MathMin(n, fftSize); i++) {
            outReal[i] = inReal[i];
        }
        
        // Perform FFT
        if(!inverse) {
            FFT(outReal, outImag);
        } else {
            IFFT(outReal, outImag);
        }
        
        return true;
    }
    
    // FFT implementation (Cooley-Tukey algorithm)
    void FFT(double& real[], double& imag[]) {
        // Bit-reversal permutation
        for(int i = 0; i < fftSize; i++) {
            int j = ReverseBits(i, (int)(MathLog(fftSize) / MathLog(2)));
            if(i < j) {
                // Swap elements
                double tempReal = real[i];
                real[i] = real[j];
                real[j] = tempReal;
                
                double tempImag = imag[i];
                imag[i] = imag[j];
                imag[j] = tempImag;
            }
        }
        
        // Cooley-Tukey FFT algorithm
        for(int s = 1; s <= (int)(MathLog(fftSize) / MathLog(2)); s++) {
            int m = 1 << s; // 2^s
            double omega_m_real = MathCos(2 * M_PI / m);
            double omega_m_imag = -MathSin(2 * M_PI / m);
            
            for(int k = 0; k < fftSize; k += m) {
                double omega_real = 1.0;
                double omega_imag = 0.0;
                
                for(int j = 0; j < m/2; j++) {
                    int index1 = k + j;
                    int index2 = k + j + m/2;
                    
                    double temp_real = omega_real * real[index2] - omega_imag * imag[index2];
                    double temp_imag = omega_real * imag[index2] + omega_imag * real[index2];
                    
                    real[index2] = real[index1] - temp_real;
                    imag[index2] = imag[index1] - temp_imag;
                    
                    real[index1] += temp_real;
                    imag[index1] += temp_imag;
                    
                    // Update omega
                    double temp = omega_real;
                    omega_real = omega_real * omega_m_real - omega_imag * omega_m_imag;
                    omega_imag = temp * omega_m_imag + omega_imag * omega_m_real;
                }
            }
        }
    }
    
    // Inverse FFT implementation
    void IFFT(double& real[], double& imag[]) {
        // Perform regular FFT
        FFT(real, imag);
        
        // Conjugate and scale
        for(int i = 0; i < fftSize; i++) {
            imag[i] = -imag[i] / fftSize;
            real[i] = real[i] / fftSize;
        }
    }
    
    // Utility function to reverse bits
    int ReverseBits(int x, int numBits) {
        int result = 0;
        for(int i = 0; i < numBits; i++) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }
    
    // Extract dominant frequencies
    void ExtractDominantFrequencies(const double &fftReal[], const double &fftImag[], double &frequencies[], double &magnitudes[], int maxFreqs = 5) {
        int n = MathMin(fftSize / 2, ArraySize(fftReal) / 2); // Only use first half (Nyquist)
        
        // Calculate magnitudes
        double mags[];
        ArrayResize(mags, n);
        
        for(int i = 0; i < n; i++) {
            mags[i] = MathSqrt(fftReal[i] * fftReal[i] + fftImag[i] * fftImag[i]);
        }
        
        // Find indices of top frequencies
        int indices[];
        ArrayResize(indices, maxFreqs);
        
        for(int i = 0; i < maxFreqs; i++) {
            int maxIndex = 0;
            double maxValue = 0;
            
            for(int j = 1; j < n; j++) {
                if(mags[j] > maxValue) {
                    maxValue = mags[j];
                    maxIndex = j;
                }
            }
            
            indices[i] = maxIndex;
            mags[maxIndex] = 0; // Remove this peak for next iteration
        }
        
        // Resize output arrays
        ArrayResize(frequencies, maxFreqs);
        ArrayResize(magnitudes, maxFreqs);
        
        // Fill output arrays
        for(int i = 0; i < maxFreqs; i++) {
            frequencies[i] = (double)indices[i] / fftSize; // Normalized frequency
            magnitudes[i] = MathSqrt(fftReal[indices[i]] * fftReal[indices[i]] + 
                                    fftImag[indices[i]] * fftImag[indices[i]]);
        }
    }
    
    // Get FFT size
    int GetFFTSize() const {
        return fftSize;
    }
};
