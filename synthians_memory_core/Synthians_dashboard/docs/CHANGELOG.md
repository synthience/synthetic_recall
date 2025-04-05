# Changelog

## [1.0.1] - 2025-04-05

### Fixed

- Added missing React imports to various components to fix "React is not defined" errors:
  - `App.tsx`
  - `DashboardShell.tsx`
  - `Sidebar.tsx`
  - `TopBar.tsx`
  - `toaster.tsx`
  - `skeleton.tsx`

- Fixed DOM nesting issues:
  - Changed nested `<a>` tags to `<div>` elements in `Sidebar.tsx` NavLink component
  - Changed nested `<a>` tags to `<div>` elements in `TopBar.tsx` Link component
  - Fixed type error in `Sidebar.tsx` by using `Boolean()` for conditional path checking

- Improved DashboardShell layout:
  - Enhanced mobile responsiveness
  - Fixed sidebar visibility in mobile and desktop views
  - Streamlined main content container structure

### Changed

- Updated `vite.config.ts` to use absolute paths for module aliases
- Removed conflicting JSX runtime options in configuration files
- Added proper cursor pointer styling to clickable elements

### Technical Details

- Path alias configuration in `vite.config.ts` was updated to avoid conflicts
- React 18 automatic JSX runtime is now properly utilized across components
- Invalid DOM nesting (nested `<a>` elements) resolved for better accessibility and standard compliance
