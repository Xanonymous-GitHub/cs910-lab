from utils.file import ls, read_file_lines_from, write_file_lines_to


def start():
    target_dir = "../dataset/Aurora"
    origin_file_names = ls(target_dir)

    cleaned_file_lines: dict[str, [str]] = {}

    for origin_file_name in origin_file_names:
        file_lines = tuple(read_file_lines_from(f"{target_dir}/{origin_file_name}"))
        file_year = origin_file_name.split('_')[2]

        for file_line in file_lines:
            # Separate the date from current line
            raw_date_str = (items := file_line.split(', '))[0]
            data_strs = items[1:]

            # check if the data includes item that is not NaN more than 40
            if len(data_strs) - data_strs.count('NaN') < 40:
                continue

            # Re-organize the date string
            year, month, day = (date_and_time := raw_date_str.split())[0].split('-')
            hour, minute, second = date_and_time[1].split(':')

            # Concatenate the year and month with the data items
            cleaned_line = ','.join([year, month, day, hour, minute, second] + data_strs)
            cleaned_file_lines.setdefault(file_year, []).append(cleaned_line)

    for file_year, cleaned_lines in cleaned_file_lines.items():
        sorted_lines = sorted(cleaned_lines, key=lambda line: ''.join(line.split(',')[:6]))
        write_file_lines_to(f"{target_dir}/cleaned_{file_year}.csv", lines=sorted_lines)


if __name__ == '__main__':
    start()
