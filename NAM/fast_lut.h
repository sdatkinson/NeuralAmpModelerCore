#include <vector>
#include <cmath>
#include <algorithm>

template <typename Func>
class FastLUT {
public:
    FastLUT(float min_x, float max_x, std::size_t size, Func f)
        : min_x_(min_x), max_x_(max_x), size_(size) {
        
        step_ = (max_x - min_x) / (size - 1);
        inv_step_ = 1.0f / step_;
        table_.reserve(size);

        for (std::size_t i = 0; i < size; ++i) {
            table_.push_back(f(min_x + i * step_));
        }
    }

    // Fast lookup with linear interpolation
    float operator()(float x) const {
        // Clamp input to range
        x = std::clamp(x, min_x_, max_x_);

        // Calculate float index
        float f_idx = (x - min_x_) * inv_step_;
        std::size_t i = static_cast<std::size_t>(f_idx);
        
        // Handle edge case at max_x_
        if (i >= size_ - 1) return table_.back();

        // Linear interpolation: y = y0 + (y1 - y0) * fractional_part
        float frac = f_idx - static_cast<float>(i);
        return table_[i] + (table_[i + 1] - table_[i]) * frac;
    }

private:
    float min_x_, max_x_, step_, inv_step_;
    size_t size_;
    std::vector<float> table_;
};
