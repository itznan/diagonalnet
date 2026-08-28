package main

// Tensor represents a flat contiguous 1D slice representation for multi-dimensional 3D tensors [C x H x W]
// to maximize CPU L1/L2 cache locality and eliminate pointer chasing.
type Tensor struct {
	Data     []float32
	Channels int
	Height   int
	Width    int
}

// NewTensor allocates a new Tensor with dimensions C x H x W
func NewTensor(c, h, w int) *Tensor {
	return &Tensor{
		Data:     make([]float32, c*h*w),
		Channels: c,
		Height:   h,
		Width:    w,
	}
}

// Index computes the contiguous 1D slice index for coordinates (c, y, x):
// Index(c, y, x) = c * (Height * Width) + y * Width + x
func (t *Tensor) Index(c, y, x int) int {
	return c*(t.Height*t.Width) + y*t.Width + x
}

// Get returns the value at coordinate (c, y, x)
func (t *Tensor) Get(c, y, x int) float32 {
	return t.Data[c*(t.Height*t.Width)+y*t.Width+x]
}

// Set stores a value at coordinate (c, y, x)
func (t *Tensor) Set(c, y, x int, val float32) {
	t.Data[c*(t.Height*t.Width)+y*t.Width+x] = val
}

// Zero resets all elements in the tensor to 0
func (t *Tensor) Zero() {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// Size returns the total number of float32 elements (C * H * W)
func (t *Tensor) Size() int {
	return len(t.Data)
}

// Clone creates an exact deep copy of the tensor
func (t *Tensor) Clone() *Tensor {
	cp := NewTensor(t.Channels, t.Height, t.Width)
	copy(cp.Data, t.Data)
	return cp
}

// Shape returns the tensor dimensions (Channels, Height, Width)
func (t *Tensor) Shape() (int, int, int) {
	return t.Channels, t.Height, t.Width
}
