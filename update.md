# ComfyUI-JoyCaption Update Log

## V1.1.0 (2025-06-05)
### Features
- Initial release of ComfyUI-JoyCaption
- Added JoyCaption node for image captioning
- Integrated memory management system
- Added caption tools for text processing

### Memory Management
- Implemented efficient memory handling for large image processing
- Added automatic memory cleanup after processing
- Optimized memory usage during batch operations
- Added memory usage monitoring

### Caption Tools
- Added Image Batch Path node (🖼️) for batch image loading
  - Support for sequential, reverse, and random image loading
  - Configurable batch size and start position
  - Automatic EXIF orientation correction
  - Support for jpg, jpeg, png, and webp formats
- Added Caption Saver node (📝) for caption management
  - Flexible output path configuration
  - Custom filename support
  - Optional image copying with captions
  - Automatic file overwrite protection
  - UTF-8 encoding support
  - Batch processing capability