
# MoBIE Plugin Setup Instructions

## Automatic Configuration (Recommended)

1. **Launch Fiji**: Run `launch_fiji.bat`
2. **Open Update Manager**: Help → Update...
3. **Manage Update Sites**: Click "Manage update sites"
4. **Enable Required Sites**: Check these update sites:
   - ✅ MoBIE
   - ✅ BigDataViewer-Playground
   - ✅ BigDataViewer-core
   - ✅ ImageJ-server
   - ✅ Fiji (should be enabled by default)

5. **Apply Changes**: Click "Apply changes" and restart Fiji

## Manual Database Installation (Already Done)

✅ MoBIE database file copied to: `fiji_new\db\db.xml.gz`
✅ Extracted XML database to: `fiji_new\db\db.xml`

## Using MoBIE

After installation, you can access MoBIE features via:
- **Plugins → MoBIE → Open MoBIE Project**
- **Plugins → BigDataViewer → Browse Current Image**
- **File → Import → Bio-Formats** (for various image formats)

## MoBIE Features for Zarr Files

- Multi-scale image viewing
- Segmentation overlays
- Table-based data exploration
- Bookmarks and annotations
- Multi-modal data visualization

## Troubleshooting

If MoBIE doesn't appear in the plugins menu:
1. Restart Fiji completely
2. Check Help → Update... → View local modifications
3. Ensure all update sites are properly enabled
4. Try Help → Refresh Menus
