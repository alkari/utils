#!/bin/bash

# This script will scan for unpartitioned physical disks and use to extend the root / lvm partition.
#
# maintained by: info@manceps.com
#
# -------------------------------------------------------------------------------------------------

for cmd in lsblk fdisk pvcreate vgscan vgchange vgextend lvextend blkid resize2fs xfs_growfs; do
    command -v $cmd >/dev/null 2>&1 || { echo "$cmd is required but not installed. Aborting."; exit 1; }
done

trap 'echo "An error occurred. Exiting."' ERR

read -p "Proceed with partitioning and extending LVM? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

set -euo pipefail

echo "Starting scan for unpartitioned drives..."

# Find all disks except loop and ram
ALL_DISKS=($(lsblk -dn -o NAME,TYPE | awk '$2=="disk"{print $1}'))

UNPARTED_DISKS=()
for disk in "${ALL_DISKS[@]}"; do
    part_count=$(lsblk -n "/dev/$disk" -o NAME,TYPE | awk '$2=="part"' | wc -l)
    if [ "$part_count" -eq 0 ]; then
        echo "Found unpartitioned disk: /dev/$disk"
        UNPARTED_DISKS+=("/dev/$disk")
    else
        echo "Disk /dev/$disk has $part_count partition(s), skipping."
    fi
done

if [ ${#UNPARTED_DISKS[@]} -eq 0 ]; then
    echo "No unpartitioned disks found. Exiting."
    exit 0
fi

ROOT_DEV=$(findmnt -n -o SOURCE /)
if [[ ! $ROOT_DEV =~ ^/dev/mapper/ ]]; then
    echo "Root device $ROOT_DEV is not an LVM logical volume. Exiting."
    exit 1
fi

# Remove '/dev/mapper/' prefix
MAPPED_NAME="${ROOT_DEV#/dev/mapper/}"

# Split only on the last dash for LV (since VG names may contain dashes)
VG="${MAPPED_NAME%-*}"
LV="${MAPPED_NAME##*-}"

# Replace double dashes with single dashes in VG and LV names
VG="${VG//--/-}"
LV="${LV//--/-}"
echo "Root filesystem is on VG: $VG, LV: $LV"

for disk in "${UNPARTED_DISKS[@]}"; do

echo "Partitioning disk $disk with a single LVM partition using fdisk..."
(
    echo o      # Create a new empty DOS partition table (MBR)
    echo n      # Add a new partition
    echo p      # Primary partition
    echo 1      # Partition number 1
    echo        # Default - first sector
    echo        # Default - last sector (use entire disk)
    echo t      # Change partition type
    echo 8e     # Set type to Linux LVM
    echo w      # Write changes
) | fdisk "$disk"
echo "Partitioned $disk. Waiting for kernel to recognize new partition..."
if command -v partprobe >/dev/null 2>&1; then
    partprobe "$disk"
else
    echo "partprobe not found, using blockdev --rereadpt and udevadm settle."
    blockdev --rereadpt "$disk" || true
    udevadm settle
fi
sleep 2

    PART="${disk}1"
    echo "Creating physical volume on $PART..."
    pvcreate "$PART"

    echo "Rescanning LVM metadata..."
    vgscan

    echo "Activating all volume groups..."
    vgchange -a y


    echo "Extending volume group $VG with $PART..."
    vgextend "$VG" "$PART"
done

echo "Extending logical volume /dev/$VG/$LV to use all free space..."
lvextend -l +100%FREE "/dev/$VG/$LV"

echo "Detecting filesystem type on /dev/$VG/$LV..."
FSTYPE=$(blkid -s TYPE -o value "/dev/$VG/$LV")
echo "Filesystem type is $FSTYPE"

if [[ "$FSTYPE" == "xfs" ]]; then
    echo "Resizing XFS filesystem..."
    xfs_growfs /
elif [[ "$FSTYPE" == "ext4" || "$FSTYPE" == "ext3" || "$FSTYPE" == "ext2" ]]; then
    echo "Resizing ext filesystem with resize2fs..."
    resize2fs "/dev/$VG/$LV"
else
    echo "Unsupported filesystem type: $FSTYPE. Please resize manually."
    exit 1
fi

echo "Filesystem resize complete. Current disk usage for /:"

df -h /
lsblk
echo "All done."

