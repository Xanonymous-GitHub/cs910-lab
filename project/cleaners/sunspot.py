from utils.file import ls, read_file_lines_from, write_file_lines_to


def start():
    target_dir = "../dataset/Sunspot"
    origin_file_names = ls(target_dir)

    for origin_file_name in origin_file_names:
        file_lines = tuple(read_file_lines_from(f"{target_dir}/{origin_file_name}"))
        cleaned_lines: [str] = []

        for file_line in file_lines:
            # Separate the date from current line
            raw_date_str = file_line[:6].strip()
            raw_data_str = file_line[6:].strip()

            # Re-organize the date string
            year_str = raw_date_str[:4].strip()
            month_str = raw_date_str[4:].strip()
            year = int(year_str)
            month = int(month_str)

            # Re-organize the data string
            data_items = raw_data_str.split()

            # Concatenate the year and month with the data items
            cleaned_line = f"{year},{month},{','.join(data_items)}"
            cleaned_lines.append(cleaned_line)

        # Write the cleaned lines to the file
        write_file_lines_to(f"{target_dir}/cleaned_{origin_file_name}", lines=cleaned_lines)


if __name__ == '__main__':
    start()
