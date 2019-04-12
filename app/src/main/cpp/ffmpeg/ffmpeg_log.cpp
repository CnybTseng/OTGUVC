
#include <stdio.h>
#include <stdbool.h>
#include "ffmpeg/ffmpeg_log.h"
#include "utils/utilbase.h"


void ffmpeg_av_log_default_callback(void* ptr, int level, const char* fmt, va_list vl)
{
    static int print_prefix = 1;
    static int count;
    static char prev[LINE_SZ];
    char part[3][LINE_SZ];
    char line[LINE_SZ];
    static int is_atty;
    int type[2];

    if (level > av_log_get_level())
        return;
    format_line(ptr, level, fmt, vl, part, sizeof(part[0]), &print_prefix, type);
    sprintf(line, "%s%s%s", part[0], part[1], part[2]);

    strcpy(prev, line);
    sanitize(part[0]);
    colored_fputs(type[0], part[0]);
    sanitize(part[1]);
    colored_fputs(type[1], part[1]);
    sanitize(part[2]);
    colored_fputs(av_clip(level >> 3, 0, 6), part[2]);
}

static void format_line(void *ptr, int level, const char *fmt, va_list vl, char part[3][LINE_SZ], int part_size, int *print_prefix, int type[2])
{
    AVClass* avc = ptr ? *(AVClass **) ptr : NULL;
    part[0][0] = part[1][0] = part[2][0] = 0;
    if(type) type[0] = type[1] = AV_CLASS_CATEGORY_NA + 16;
    if (*print_prefix && avc) {
        if (avc->parent_log_context_offset) {
            AVClass** parent = *(AVClass ***) (((uint8_t *) ptr) +
                                   avc->parent_log_context_offset);
            if (parent && *parent) {
                sprintf(part[0], "[%s @ %p] ",
                         (*parent)->item_name(parent), parent);
                if(type) type[0] = get_category(parent);
            }
        }
        sprintf(part[1], "[%s @ %p] ",
                 avc->item_name(ptr), ptr);
        if(type) type[1] = get_category(ptr);
    }

    vsprintf(part[2], fmt, vl);

    if(*part[0] || *part[1] || *part[2])
        *print_prefix = strlen(part[2]) && part[2][strlen(part[2]) - 1] == '\n';
}

static void sanitize(char *line){
    while(*line){
        if(*line < 0x08 || (*line > 0x0D && *line < 0x20))
            *line='?';
        line++;
    }
}

static int get_category(void *ptr){
    AVClass *avc = *(AVClass **) ptr;
    if(    !avc
        || (avc->version&0xFF)<100
        ||  avc->version < (51 << 16 | 59 << 8)
        ||  avc->category >= AV_CLASS_CATEGORY_NB) return AV_CLASS_CATEGORY_NA + 16;

    if(avc->get_category)
        return avc->get_category(ptr) + 16;

    return avc->category + 16;
}

static void colored_fputs(int level, const char *str) {
    LOGV("%s\n",str);
}
