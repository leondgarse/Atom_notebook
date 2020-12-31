#!/bin/bash

DIST_ROOT_MOUNT_POINT='/tmp/mount_point_for_dist_root'
DIST_HOME_MOUNT_POINT=$DIST_ROOT_MOUNT_POINT'/home'
EXCLUDE_FILE='./rsync_excludes_file_list'
SYS_PATH_EXCLUDED='proc sys tmp mnt media cdrom media/cdrom0 snap'

# Parsing arguments
if [ $# -ge 6 ]; then
    # Restore mode
    echo "Restore from a squashfs file."
    WORK_MODE="RESTORE"
    SOURCE_SQUASH_PATH=$1
    DIST_ROOT_PATH=$2
    DIST_HOME_PATH=$3
    DIST_SWAP_PATH=$4
    DIST_GRUB=$5
    HOST_NAME=$6
    SOURCE_SYSTEM_PATH='/tmp/mount_point_for_source_squash'

    echo "SOURCE_SQUASH_PATH = $SOURCE_SQUASH_PATH"
    echo "DIST_ROOT_PATH = $DIST_ROOT_PATH"
    echo "DIST_HOME_PATH = $DIST_HOME_PATH"
    echo "DIST_SWAP_PATH = $DIST_SWAP_PATH"
    echo "DIST_GRUB = $DIST_GRUB"
    echo "HOST_NAME = $HOST_NAME"
elif [ $# -eq 5 ]; then
    # Clone mode
    echo "Clone current system."
    WORK_MODE="CLONE"
    DIST_ROOT_PATH=$1
    DIST_HOME_PATH=$2
    DIST_SWAP_PATH=$3
    DIST_GRUB=$4
    HOST_NAME=$5
    SOURCE_SYSTEM_PATH='/'

    echo "DIST_ROOT_PATH = $DIST_ROOT_PATH"
    echo "DIST_HOME_PATH = $DIST_HOME_PATH"
    echo "DIST_SWAP_PATH = $DIST_SWAP_PATH"
    echo "HOST_NAME = $HOST_NAME"
elif [ $# -eq 1 ]; then
    # Backup mode
    echo "Backup current system to a squashfs file."
    WORK_MODE="BACKUP"
    SQUASHFS_BACKUP_TO=$1
    TEMP_SYSTEM_DIR='/tmp/temp_system_dir'

    echo "SQUASHFS_BACKUP_TO = $SQUASHFS_BACKUP_TO"
else
    echo "Restore Usage: `echo $0 | xargs basename` <source squash path> <dist root disk> <dist home disk> <dist swap disk> <dist grub disk> <host name>"
    echo "Clone   Usage: `echo $0 | xargs basename` <dist root disk> <dist home disk> <dist swap disk> <dist grub disk> <host name>"
    echo "Backup  Usage: `echo $0 | xargs basename` <squashfs file backup to>"
    exit
fi

# Function to generate exclude file list
function generate_exclude_list_base {
    printf "
/snap
/cdrom
/proc
/sys
/tmp
/mnt
/media
/boot/grub
/etc/fstab
/etc/mtab
/etc/blkid.tab
/etc/udev/rules.d/70-persistent-net.rules
/host
/lost+found
/home/lost+found
/root/.gvfs
/var/crash
/var/log
/swapfile
`ls -1 /home/*/.gvfs 2>/dev/null`
`ls -1 /lib/modules/\`uname -r\`/volatile/ 2>/dev/null`
`ls -1 /var/cache/apt/archives/partial/ 2>/dev/null`
`ls -d1 /var/log/journal/* 2>/dev/null`
`find /run/user/* -maxdepth 1 -name gvfs 2>/dev/null`
`find /var -type s 2>/dev/null`
`find /run -type s 2>/dev/null`
" > $EXCLUDE_FILE

    # This may contain special characters for printf
    ls -1 /var/cache/apt/archives/*.deb 2>/dev/null >> $EXCLUDE_FILE
}

function generate_exclude_list_addition {
    generate_exclude_list_base
    printf "
/home
" >> $EXCLUDE_FILE
}

function clean_resource_and_exit {
    echo $1
    umount $DIST_HOME_MOUNT_POINT 2>/dev/null
    sleep 1
    umount $DIST_ROOT_MOUNT_POINT 2>/dev/null
    sleep 1
    if [ $? -eq 0 ]; then
        echo "temp mount path: "$DIST_HOME_MOUNT_POINT" "$DIST_ROOT_MOUNT_POINT" ,remove at your choice"
        # rm $DIST_HOME_MOUNT_POINT -r
        # rm $DIST_ROOT_MOUNT_POINT -r
    fi

    if [ $WORK_MODE = "RESTORE" ]; then
        umount $SOURCE_SYSTEM_PATH 2>/dev/null
        # rm $SOURCE_SYSTEM_PATH -rf
    fi

    rm $EXCLUDE_FILE -f

    exit
}

function chroot_command {
    mount --bind /proc $1/proc
    mount --bind /dev $1/dev
    mount --bind /sys $1/sys
    chroot $*
    umount $1/proc
    umount $1/dev
    umount $1/sys
}

# generate_exclude_list
# exit

# Check if it's run by root
USER_NAME=`whoami`
echo "USER_NAME = $USER_NAME"
if [ $USER_NAME != "root" ]; then
    echo "Should be run as root!"
    exit
fi

DATE_START_S=`date +%s`

if [ $WORK_MODE != "BACKUP" ]; then
    # Clone and Restore mode
    # Format disks, but not force
    umount $DIST_HOME_PATH
    umount $DIST_ROOT_PATH

    mkfs.ext4 $DIST_ROOT_PATH
    mkfs.ext4 $DIST_HOME_PATH # Select 'n' if you don't want to format HOME disk.
    mkswap $DIST_SWAP_PATH

    # if [ $? -ne 0 ]; then echo "mkfs error"; exit; fi

    # Mount dist disks
    mkdir -p $DIST_ROOT_MOUNT_POINT && \
        mount $DIST_ROOT_PATH $DIST_ROOT_MOUNT_POINT && \
        mkdir -p $DIST_HOME_MOUNT_POINT && \
        mount $DIST_HOME_PATH $DIST_HOME_MOUNT_POINT

    if [ $? -ne 0 ]; then clean_resource_and_exit "mount dist disks error"; fi

    if [ $WORK_MODE = "RESTORE" ]; then
        # It's Restore mode, mount source fs
        mkdir -p $SOURCE_SYSTEM_PATH && \
            mount "$SOURCE_SQUASH_PATH" $SOURCE_SYSTEM_PATH -o loop

        if [ $? -ne 0 ]; then clean_resource_and_exit "mount source squashfs error"; fi
    fi

    # rsync, need an exclude file list
    generate_exclude_list_base
    # exit
    rsync -av --exclude-from=$EXCLUDE_FILE $SOURCE_SYSTEM_PATH/ $DIST_ROOT_MOUNT_POINT
    # rsync -av \
    #     --exclude "/lost+found" \
    #     --exclude "/*/lost+found" \
    #     --exclude "/lib/modules/*/volatile/*" \
    #     $SOURCE_SYSTEM_PATH/ $DIST_ROOT_MOUNT_POINT
    if [ $? -ne 0 ]; then
        read -p "rsync error, proceed anyway? (Y/n):" RESULT
        if [[ ! $RESULT =~ ^[yY] ]]; then
            clean_resource_and_exit "rsync error, exit"
        else
            echo 'OK, so we GO ON!'
        fi
    fi

    # Create excluded system path
    cd $DIST_ROOT_MOUNT_POINT && \
        mkdir -p $SYS_PATH_EXCLUDED && \
        chmod 1777 tmp

    if [ $? -ne 0 ]; then clean_resource_and_exit "mkdir error"; fi

    # Create fstab and mtab
    DIST_ROOT_UUID=`blkid $DIST_ROOT_PATH -s UUID -o value`
    DIST_HOME_UUID=`blkid $DIST_HOME_PATH -s UUID -o value`
    DIST_SWAP_UUID=`blkid $DIST_SWAP_PATH -s UUID -o value`

    mkdir -p etc
    printf "
# /etc/fstab: static file system information.
#
# Use 'blkid -o value -s UUID' to print the universally unique identifier
# for a device; this may be used with UUID= as a more robust way to name
# devices that works even if disks are added and removed. See fstab(5).
#
# <file system> <mount point>   <type>  <options>       <dump>  <pass>
proc            /proc           proc    nodev,noexec,nosuid 0       0
#$DIST_ROOT_PATH
UUID=$DIST_ROOT_UUID      /      ext4      errors=remount-ro      0      1
#$DIST_HOME_PATH
UUID=$DIST_HOME_UUID      /home      ext4      defaults      0      2
#$DIST_SWAP_PATH
UUID=$DIST_SWAP_UUID       none            swap    sw              0       0
    " > etc/fstab && \
        touch etc/mtab

    if [ $? -ne 0 ]; then clean_resource_and_exit "Create fstab error"; fi

    # Update hostname
    OLD_HOSTNAME=`cat etc/hostname`
    echo $HOST_NAME > etc/hostname && \
        sed -i 's/^127.0.1.1\s*'$OLD_HOSTNAME'/127.0.1.1\t'$HOST_NAME'/' etc/hosts

    if [ $? -ne 0 ]; then clean_resource_and_exit "Update hostname error"; fi

    # Install grub, grub-install may fail here
    # [???]
    # grub-install --boot-directory=$DIST_ROOT_MOUNT_POINT/boot $DIST_ROOT_PATH && \
    # update-grub -o $DIST_ROOT_MOUNT_POINT/boot/grub/grub.cfg
    # chroot_command $DIST_ROOT_MOUNT_POINT grub-install --boot-directory=$DIST_ROOT_MOUNT_POINT/boot ${DIST_ROOT_PATH:0:-1} && \
    grub-install --boot-directory=$DIST_ROOT_MOUNT_POINT/boot --target=i386-pc $DIST_GRUB && \
    chroot_command $DIST_ROOT_MOUNT_POINT update-grub

    if [ $? -ne 0 ]; then clean_resource_and_exit "Install grub error"; fi

    cd -
    clean_resource_and_exit "Done!"
else
    # Backup mode
    generate_exclude_list_addition
    # generate_exclude_list_base
    if [ -e $SQUASHFS_BACKUP_TO ]; then
        read -p "Target file $SQUASHFS_BACKUP_TO exist, overrite it? (Y/n):" RESULT
        if [[ ! $RESULT =~ ^[yY] ]]; then echo 'Please give another name then'; exit; fi

        rm -f $SQUASHFS_BACKUP_TO
    fi

    mksquashfs / "$SQUASHFS_BACKUP_TO" -no-duplicates -ef $EXCLUDE_FILE -e "$SQUASHFS_BACKUP_TO"
    if [ $? -ne 0 ]; then echo "mksquashfs error"; exit; fi

	# Error here: Failed to read existing filesystem - will not overwrite - ABORTING!
    # mkdir -p $TEMP_SYSTEM_DIR && \
    #     cd $TEMP_SYSTEM_DIR && \
    #     mkdir -p $SYS_PATH_EXCLUDED && \
    #     chmod 1777 tmp
    # if [ $? -ne 0 ]; then echo "make system dirs error"; exit; fi

    # mksquashfs $TEMP_SYSTEM_DIR "$SQUASHFS_BACKUP_TO" -no-duplicates
    # if [ $? -ne 0 ]; then echo "mksquashfs error"; exit; fi

    # cd -
    # rm $TEMP_SYSTEM_DIR -rf
    rm $EXCLUDE_FILE -f

    BACKUP_SIZE=`ls $SQUASHFS_BACKUP_TO -lh | cut -d ' ' -f 5`
    echo "Backup size = $BACKUP_SIZE"
fi

DATE_END_S=`date +%s`
TOTAL_MINUTE=$(expr \( $DATE_END_S - $DATE_START_S \) / 60)
TOTAL_SECOND=$(expr \( $DATE_END_S - $DATE_START_S \) % 60)
TOTAL_TIME=${TOTAL_MINUTE}"m"${TOTAL_SECOND}"s"
echo "Total time = $TOTAL_TIME"
